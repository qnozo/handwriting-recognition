#include <Arduino.h>

#include "FS.h"
#include <TFT_eSPI.h> // Hardware-specific library
#include <SPI.h>
// #include <TensorFlowLite_ESP32.h>

#include "model.h"
#include "model_settings.h"


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
TFT_eSPI tft = TFT_eSPI(); // Invoke custom library

#define CALIBRATION_FILE "/TouchCalData3"
#define REPEAT_CAL false

#define PIXEL_TIMEOUT 100   // 100ms Time-out between pixel requests
#define START_TIMEOUT 10000 // 10s Maximum time to wait at start transfer

#define BITS_PER_PIXEL 24 // 24 for RGB colour format, 16 for 565 colour format

// File names must be alpha-numeric characters (0-9, a-z, A-Z) or "/" underscore "_"
// other ascii characters are stripped out by client, including / generates
// sub-directories
#define DEFAULT_FILENAME "tft_screenshots/screenshot" // In case none is specified
#define FILE_TYPE "png"                               // jpg, bmp, png, tif are valid

// Filename extension
// '#' = add incrementing number, '@' = add timestamp, '%' add millis() timestamp,
// '*' = add nothing
// '@' and '%' will generate new unique filenames, so beware of cluttering up your
// hard drive with lots of images! The PC client sketch is set to limit the number of
// saved images to 1000 and will then prompt for a restart.
#define FILE_EXT '#'

// Number of pixels to send in a burst (minimum of 1), no benefit above 8
// NPIXELS values and render times:
// NPIXELS 1 = use readPixel() = >5s and 16 bit pixels only
// NPIXELS >1 using rectRead() 2 = 1.75s, 4 = 1.68s, 8 = 1.67s
#define NPIXELS 1 // Must be integer division of both TFT width and TFT height

// touch box settings:
#define BOX_X 0
#define BOX_Y 0
#define BOX_W 240
#define BOX_H 180

#define SIZE 6
//--------------------

// buttons:
#define BUTTONS_X 40
#define BUTTONS_Y 289
#define BUTTONS_W 62
#define BUTTONS_H 30
#define BUTTONS_SPACING_X 18 // X and Y gap
#define BUTTONS_SPACING_Y 25
#define BUTTONS_TEXTSIZE 1
char buttonsLabel[4][8] = {"Clear", "Predict", "Send", "Del"};
uint16_t buttonsColor[4] = {TFT_RED, TFT_LIGHTGREY, TFT_GREEN, TFT_RED};

TFT_eSPI_Button buttons[4];

//---------
#define TEXT_X 5
#define TEXT_Y 220
#define TEXT_WIDTH 230
#define TEXT_HEIGHT 40
// uint8_t image[BOX_H * BOX_W];
uint8_t *image;
#define TO_SERIAL false
char text[256] = "";
namespace
{
    tflite::ErrorReporter *error_reporter = nullptr;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    int inference_count = 0;

    constexpr int kTensorArenaSize = 1024 * 26;
    static uint8_t *tensor_arena;

} // namespace

void touch_calibrate();
bool contains(int16_t x, int16_t y);
void get_image();
void resize(uint8_t *image, uint8_t *resized_image, uint8_t w1, uint8_t h1, uint8_t w2, uint8_t h2);
void image_to_serial(uint8_t *resized_image);
void sendParameters();
char predict(uint8_t *resized_image);

void setup()
{
    Serial.begin(921600);
    tft.init();

    // Set the rotation before we calibrate
    tft.setRotation(0);

    // call screen calibration
    touch_calibrate();

    // clear screen
    tft.fillScreen(TFT_CYAN);

    tft.fillRect(BOX_X, BOX_Y, BOX_W, BOX_H, TFT_WHITE);


    tft.fillRect(TEXT_X, TEXT_Y, TEXT_WIDTH, TEXT_HEIGHT, TFT_WHITE);
    tft.drawRect(TEXT_X, TEXT_Y, TEXT_WIDTH, TEXT_HEIGHT, TFT_LIGHTGREY);

    for (uint8_t i = 0; i < 3; i++)
    {
        buttons[i].initButton(&tft, BUTTONS_X + i * (BUTTONS_W + BUTTONS_SPACING_X), BUTTONS_Y, BUTTONS_W, BUTTONS_H, TFT_BLACK, buttonsColor[i], TFT_BLACK, buttonsLabel[i], BUTTONS_TEXTSIZE);
        buttons[i].drawButton();
    }
    buttons[3].initButton(&tft, TEXT_WIDTH - 30, TEXT_Y + 20, BUTTONS_W, BUTTONS_H, TFT_BLACK, buttonsColor[3], TFT_BLACK, buttonsLabel[3], BUTTONS_TEXTSIZE);
    buttons[3].drawButton();


    

    image = (uint8_t *)heap_caps_malloc(BOX_W * BOX_H, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    // tensorflow lite setup
    // tflite::InitializeTarget();

    if (tensor_arena == NULL)
    {
        tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (tensor_arena == NULL)
    {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return;
    }
    // Set up logging. Google style is to avoid globals or statics because of
    // lifetime uncertainty, but since this has a trivial destructor it's okay.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        error_reporter->Report("Model provided is schema version %d not equal "
                               "to supported version %d.",
                               model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        error_reporter->Report("AllocateTensors() failed");
        return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void loop()
{

    uint16_t x, y;
    // See if there's any touch data for us
    if (tft.getTouch(&x, &y))
    {
        if (contains(x, y))
            tft.fillCircle(x, y, SIZE, TFT_BLACK);

        for (uint8_t b = 0; b < 4; b++)
        {
            if (buttons[b].contains(x, y))
                buttons[b].press(true);
            else
                buttons[b].press(false);
        }

        for (uint8_t b = 0; b < 4; b++)
        {
            if (buttons[b].justReleased())
                buttons[b].drawButton();
            if (buttons[b].justPressed())
            {
                buttons[b].drawButton(true);
                switch (b)
                {
                case 0:
                    // clear screen
                    tft.fillRect(BOX_X, BOX_Y, BOX_W, BOX_H, TFT_WHITE);
                    break;
                case 1:
                    {
                    // get_image

                    get_image();

                    uint8_t resized_image[kNumRows * kNumCols];
                    resize(image, resized_image, BOX_W, BOX_H, kNumCols, kNumRows);
                    int len = strlen(text);
                    text[len] = predict(resized_image);
                    text[len + 1] = '\0';
                    
                    // tft.fillRect(TEXT_X, TEXT_Y, TEXT_WIDTH, TEXT_HEIGHT, TFT_WHITE);
                    tft.setTextColor(TFT_BLACK, TFT_WHITE);
                    tft.drawString(text, TEXT_X + 10, TEXT_Y + 10, 4);

                    tft.fillRect(BOX_X, BOX_Y, BOX_W, BOX_H, TFT_WHITE);
                    // debug
                    if (TO_SERIAL)
                        image_to_serial(resized_image);
                    break;
                    }
                case 2:
                    // send

                    break;
                case 3: 
                    // clear text
                    tft.fillRect(TEXT_X, TEXT_Y, TEXT_WIDTH, TEXT_HEIGHT, TFT_WHITE);
                    tft.drawRect(TEXT_X, TEXT_Y, TEXT_WIDTH, TEXT_HEIGHT, TFT_LIGHTGREY);
                    int len = strlen(text);
                    text[len-1] = '\0';
                    tft.setTextColor(TFT_BLACK, TFT_WHITE);
                    tft.drawString(text, TEXT_X + 10, TEXT_Y + 10, 4);
                    buttons[3].drawButton(true);
                    break;
                }
                
            }
        }
    }
}

void touch_calibrate()
{
    uint16_t calData[5];
    uint8_t calDataOK = 0;

    // check file system exists
    if (!SPIFFS.begin())
    {
        Serial.println("Formatting file system");
        SPIFFS.format();
        SPIFFS.begin();
    }

    // check if calibration file exists and size is correct
    if (SPIFFS.exists(CALIBRATION_FILE))
    {
        if (REPEAT_CAL)
        {
            // Delete if we want to re-calibrate
            SPIFFS.remove(CALIBRATION_FILE);
        }
        else
        {
            File f = SPIFFS.open(CALIBRATION_FILE, "r");
            if (f)
            {
                if (f.readBytes((char *)calData, 14) == 14)
                    calDataOK = 1;
                f.close();
            }
        }
    }

    if (calDataOK && !REPEAT_CAL)
    {
        // calibration data valid
        tft.setTouch(calData);
    }
    else
    {
        // data not valid so recalibrate
        tft.fillScreen(TFT_BLACK);
        tft.setCursor(20, 0);
        tft.setTextFont(2);
        tft.setTextSize(1);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);

        tft.println("Touch corners as indicated");

        tft.setTextFont(1);
        tft.println();

        if (REPEAT_CAL)
        {
            tft.setTextColor(TFT_RED, TFT_BLACK);
            tft.println("Set REPEAT_CAL to false to stop this running again!");
        }

        tft.calibrateTouch(calData, TFT_MAGENTA, TFT_BLACK, 15);

        tft.setTextColor(TFT_GREEN, TFT_BLACK);
        tft.println("Calibration complete!");

        // store data
        File f = SPIFFS.open(CALIBRATION_FILE, "w");
        if (f)
        {
            f.write((const unsigned char *)calData, 14);
            f.close();
        }
    }
}

bool contains(int16_t x, int16_t y)
{
    return ((x >= BOX_X) && (x < (BOX_X + BOX_W)) &&
            (y >= BOX_Y) && (y < (BOX_Y + BOX_H)));
}

void get_image()
{
    
    for (uint32_t y = 0; y < BOX_H; y++)
    {
        for (uint32_t x = 0; x < BOX_W; x++)
        {
         

            uint16_t c = tft.readPixel(x, y);
            uint8_t gray = ((c >> 8) * 0.5) + ((c & 0xFF) * 0.5);

            // uint8 color[2];
            image[(y * BOX_W) + x] = gray / 255.0f;
        }
    }
}

void image_to_serial(uint8_t *resized_image)
{
    uint32_t clearTime = millis() + 50;
    while (millis() < clearTime && Serial.read() >= 0)
        delay(0); // Equivalent to yield() for ESP8266;

    bool wait = true;
    uint32_t lastCmdTime = millis(); // Initialise start of command time-out

    // Wait for the starting flag with a start time-out
    while (wait)
    {
        delay(0); // Equivalent to yield() for ESP8266;
        // Check serial buffer
        if (Serial.available() > 0)
        {
            // Read the command byte
            uint8_t cmd = Serial.read();
            // If it is 'S' (start command) then clear the serial buffer for 100ms and stop waiting
            if (cmd == 'S')
            {
                // Precautionary receive buffer garbage flush for 50ms
                clearTime = millis() + 50;
                while (millis() < clearTime && Serial.read() >= 0)
                    delay(0); // Equivalent to yield() for ESP8266;

                wait = false;           // No need to wait anymore
                lastCmdTime = millis(); // Set last received command time

                // Send screen size etc using a simple header with delimiters for client checks
                sendParameters();
            }
        }
        else
        {
            // Check for time-out
            if (millis() > lastCmdTime + START_TIMEOUT)
                return;
        }
    }
    for (uint32_t i = 0; i < kNumRows * kNumCols; i++)
    {

        delay(0);
        while (Serial.available() == 0)
        {
            if (millis() > lastCmdTime + PIXEL_TIMEOUT)
                return;
            delay(0); // Equivalent to yield() for ESP8266;
        }

        lastCmdTime = millis();

        uint8_t color[2];
        color[0] = resized_image[i] * 255;
        color[1] = resized_image[i] * 255;

        Serial.write(color, 2);
    }
    Serial.flush();
    tft.fillRect(BOX_X, BOX_Y, BOX_W, BOX_H, TFT_WHITE);
}

void resize(uint8_t *image, uint8_t *resized_image, uint8_t w1, uint8_t h1, uint8_t w2, uint8_t h2)
{
    uint32_t x_ratio = ((w1 << 16) / w2) + 1;
    uint32_t y_ratio = ((h1 << 16) / h2) + 1;
    uint32_t x = 0, y = 0;
    for (uint32_t i = 0; i < h2; i++)
    {
    for (uint32_t j = 0; j < w2; j++)
        {
    
        
            x = ((j * x_ratio) >> 16);
            y = ((i * y_ratio) >> 16);
           

            resized_image[(i * w2) + j] = image[(y * w1) + x];
        }
    }
}

void sendParameters()
{
    Serial.write('W'); // Width
    Serial.write(kNumCols >> 8);
    Serial.write(kNumCols & 0xFF);

    Serial.write('H'); // Height
    Serial.write(kNumRows >> 8);
    Serial.write(kNumRows & 0xFF);

    Serial.write('Y'); // Bits per pixel (16 or 24)
    if (NPIXELS > 1)
        Serial.write(BITS_PER_PIXEL);
    else
        Serial.write(16); // readPixel() only provides 16 bit values

    Serial.write('?'); // Filename next
    Serial.print("tft_screenshots/screenshot");

    Serial.write('.'); // End of filename marker

    Serial.write(FILE_EXT); // Filename extension identifier

    Serial.write(*FILE_TYPE); // First character defines file type j,b,p,t
}

char predict(uint8_t *resized_image)
{

    for (uint32_t i = 0; i < kNumRows; i++)
    {
        for (uint32_t j = 0; j < kNumCols; j++)
        {
            
            input->data.uint8[i * kNumCols + j] = tflite::FloatToQuantizedType<uint8_t>(
                resized_image[(i * kNumCols) + j], input->params.scale, input->params.zero_point);

        }
      
    }


    if (kTfLiteOk != interpreter->Invoke())
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");

        return (char)0;
    }

    // Get the output tensor
    TfLiteTensor *output_tensor = interpreter->output(0);
    uint8_t index = 0;
    float max = 0;
    for (uint8_t i = 0; i < kCategoryCount; i++)
    {
        float result = (output_tensor->data.uint8[i] - output->params.zero_point) * output->params.scale;
        // float result = output_tensor->data.uint8[i];
        if (result > max)
        {
            max = result;
            index = i;
        }

    }

    return kCategoryLabels[index];
}
