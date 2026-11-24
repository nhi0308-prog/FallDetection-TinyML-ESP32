#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <math.h>
#include <string.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "kfall_ai.h"
#include "kfall_model.h"

namespace {
alignas(16) uint8_t arena[96 * 1024];
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
}

bool kfallAiBegin() {
    const tflite::Model *model = tflite::GetModel(kFallModel);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("# AI schema mismatch");
        return false;
    }

    static tflite::MicroMutableOpResolver<7> ops;
    if (ops.AddExpandDims() != kTfLiteOk ||
        ops.AddConv2D() != kTfLiteOk ||
        ops.AddReshape() != kTfLiteOk ||
        ops.AddMaxPool2D() != kTfLiteOk ||
        ops.AddMean() != kTfLiteOk ||
        ops.AddFullyConnected() != kTfLiteOk ||
        ops.AddLogistic() != kTfLiteOk) {
        Serial.println("# AI operator registration failed");
        return false;
    }

    static tflite::MicroInterpreter engine(model, ops, arena, sizeof(arena));
    if (engine.AllocateTensors() != kTfLiteOk) {
        Serial.println("# AI AllocateTensors failed");
        return false;
    }

    input = engine.input(0);
    output = engine.output(0);
    if (!input || !output ||
        input->type != kTfLiteInt8 || input->bytes != 600 ||
        input->dims->size != 3 || input->dims->data[0] != 1 ||
        input->dims->data[1] != 100 || input->dims->data[2] != 6 ||
        output->type != kTfLiteInt8 || output->bytes != 1 ||
        input->params.zero_point != -1 ||
        fabsf(input->params.scale - 0.16863128542900085f) > 1e-7f ||
        output->params.zero_point != -128 ||
        fabsf(output->params.scale - 1.0f / 256.0f) > 1e-8f) {
        Serial.println("# AI tensor shape/type/quantization mismatch");
        return false;
    }

    interpreter = &engine;
    return true;
}

bool kfallAiPredict(const int8_t *values, size_t bytes, int &score) {
    if (!interpreter || bytes != input->bytes) return false;
    memcpy(input->data.int8, values, bytes);
    if (interpreter->Invoke() != kTfLiteOk) return false;
    score = static_cast<int>(output->data.int8[0]);
    return true;
}
