#pragma once
#include <stddef.h>
#include <stdint.h>

bool kfallAiBegin();
bool kfallAiPredict(const int8_t *values, size_t bytes, int &score);
