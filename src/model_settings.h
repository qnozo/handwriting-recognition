# ifndef HANDWRITTEN_DIGIT_MODEL_SETTINGS_H_
# define HANDWRITTEN_DIGIT_MODEL_SETTINGS_H_

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

constexpr int kNumCols = 30;
constexpr int kNumRows = 23;

constexpr int kMaxImageSize = kNumCols * kNumRows;

constexpr int kCategoryCount = 62;

extern const char kCategoryLabels[kCategoryCount];

#endif // HANDWRITTEN_DIGIT_MODEL_SETTINGS_H_