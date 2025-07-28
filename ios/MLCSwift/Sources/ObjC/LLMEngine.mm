//
//  LLMEngine.mm
//  LLMEngine
//
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <os/proc.h>

#include "LLMEngine.h"

#define TVM_USE_LIBBACKTRACE 0
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/ndarray.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/int_tuple.h>

using namespace tvm::ffi;
using namespace tvm::runtime;
using tvm::ffi::Function;
using tvm::ffi::String;
using tvm::ffi::TypedFunction;

@implementation JSONFFIEngine {
  // Internal c++ classes
  // internal module backed by JSON FFI
  Module json_ffi_engine_;
  // member functions
  Function init_background_engine_func_;
  Function unload_func_;
  Function reload_func_;
  Function reset_func_;
  Function chat_completion_func_;
  Function abort_func_;
  Function run_background_loop_func_;
  Function run_background_stream_back_loop_func_;
  Function exit_background_loop_func_;
  Any tokenizer_;
  Function tokenizer_constructor;
  Function tokenizer_encode_func_;
  Function embed_func_;
}

- (instancetype)init {
  if (self = [super init]) {
    // load chat module
    Function f_json_ffi_create = Function::GetGlobalRequired("mlc.json_ffi.CreateJSONFFIEngine");
    json_ffi_engine_ = f_json_ffi_create().cast<Module>();
    init_background_engine_func_ = json_ffi_engine_->GetFunction("init_background_engine");
    reload_func_ = json_ffi_engine_->GetFunction("reload");
    unload_func_ = json_ffi_engine_->GetFunction("unload");
    reset_func_ = json_ffi_engine_->GetFunction("reset");
    chat_completion_func_ = json_ffi_engine_->GetFunction("chat_completion");
    abort_func_ = json_ffi_engine_->GetFunction("abort");
    run_background_loop_func_ = json_ffi_engine_->GetFunction("run_background_loop");
    run_background_stream_back_loop_func_ =
        json_ffi_engine_->GetFunction("run_background_stream_back_loop");
    exit_background_loop_func_ = json_ffi_engine_->GetFunction("exit_background_loop");
    embed_func_ = json_ffi_engine_->GetFunction("embed");
    tokenizer_constructor = Function::GetGlobalRequired("mlc.tokenizers.Tokenizer");
    tokenizer_encode_func_ = Function::GetGlobal("mlc.tokenizers.TokenizerEncode").value();
    
    ICHECK(init_background_engine_func_ != nullptr);
    ICHECK(reload_func_ != nullptr);
    ICHECK(unload_func_ != nullptr);
    ICHECK(reset_func_ != nullptr);
    ICHECK(chat_completion_func_ != nullptr);
    ICHECK(abort_func_ != nullptr);
    ICHECK(run_background_loop_func_ != nullptr);
    ICHECK(run_background_stream_back_loop_func_ != nullptr);
    ICHECK(exit_background_loop_func_ != nullptr);
  }
  return self;
}

- (void)initBackgroundEngine:(void (^)(NSString*))streamCallback {
  TypedFunction<void(String)> internal_stream_callback([streamCallback](String value) {
    streamCallback([NSString stringWithUTF8String:value.c_str()]);
  });
  int device_type = kDLMetal;
  int device_id = 0;
  init_background_engine_func_(device_type, device_id, internal_stream_callback);
}

- (void)reload:(NSString*)engineConfigJson {
  std::string engine_config = engineConfigJson.UTF8String;
  reload_func_(engine_config);
}

- (void)unload {
  unload_func_();
}

- (void)reset {
  reset_func_();
}

- (void) loadTokenizer: (NSString *)model {
    // --- Parse model path from JSON ---
    /*auto all_funcs = Function::ListGlobalNames();
    for (auto name : all_funcs) {
        std::cout << name << std::endl;
    }*/
    std::string modelPath = model.UTF8String;
    tokenizer_ = tokenizer_constructor(modelPath);

}

- (void)chatCompletion:(NSString*)requestJSON requestID:(NSString*)requestID {
  std::string request_json = requestJSON.UTF8String;
  std::string request_id = requestID.UTF8String;
  chat_completion_func_(request_json, request_id);
}

- (NSArray<NSNumber *> *)tokenize:(NSString*)text {
    
  std::string input = [text UTF8String];

  // Call the tokenizer encode function
  Any result = tokenizer_encode_func_(tokenizer_, input);

  // Cast to Shape
  Shape tokens = result.cast<Shape>();
  NSMutableArray<NSNumber *> *output = [NSMutableArray arrayWithCapacity:tokens.size()];
  for (int64_t id : tokens) {
    [output addObject:@(id)];
  }
  return output;
}

- (NSArray<NSNumber *> *)embedFromTokenIds:(NSArray<NSNumber *> *)tokens withLib: (NSString *) modelLib {
  std::vector<int64_t> token_vec;
  for (NSNumber *num in tokens) {
    token_vec.push_back(num.longLongValue);
  }
  IntTuple token_ids(token_vec);
  std::string modelLib_input = [modelLib UTF8String];
  NDArray embedding = embed_func_(token_ids, modelLib_input).cast<NDArray>();

  CHECK_EQ(embedding->dtype.code, kDLFloat);
  //NSLog(@"Embedding dtype: code=%d bits=%d lanes=%d", embedding->dtype.code, embedding->dtype.bits, embedding->dtype.lanes);

  // Compute total length
  int64_t len = 1;
  for (int i = 0; i < embedding->ndim; ++i) {
    len *= embedding->shape[i];
  }

  NSMutableArray<NSNumber *> *output = [NSMutableArray arrayWithCapacity:(NSUInteger)len];

  if (embedding->dtype.bits == 32) {
    const float* data = static_cast<const float*>(embedding->data);
    for (int64_t i = 0; i < len; ++i) {
      [output addObject:@(data[i])];
    }
  } else if (embedding->dtype.bits == 16) {
    const uint16_t* f16_data = static_cast<const uint16_t*>(embedding->data);
    for (int64_t i = 0; i < len; ++i) {
      uint16_t h = f16_data[i];
      uint16_t h_exp = (h & 0x7C00u) >> 10;
      uint16_t h_sig = (h & 0x03FFu);
      uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
      uint32_t exp, sig;

      if (h_exp == 0) {
        if (h_sig == 0) {
          exp = 0;
          sig = 0;
        } else {
          h_exp = 1;
          while ((h_sig & 0x0400u) == 0) {
            h_sig <<= 1;
            h_exp--;
          }
          h_sig &= 0x03FFu;
          exp = (uint32_t)(127 - 15 - h_exp + 1);
          sig = ((uint32_t)h_sig) << 13;
        }
      } else if (h_exp == 0x1F) {
        exp = 255;
        sig = (uint32_t)h_sig << 13;
      } else {
        exp = (uint32_t)(h_exp - 15 + 127);
        sig = (uint32_t)h_sig << 13;
      }

      uint32_t f32 = sign | (exp << 23) | sig;
      float result;
      memcpy(&result, &f32, sizeof(result));
      [output addObject:@(result)];
    }
  } else {
    LOG(FATAL) << "Unsupported embedding dtype bits: " << embedding->dtype.bits;
  }

  return output;
}

- (void)abort:(NSString*)requestID {
  std::string request_id = requestID.UTF8String;
  abort_func_(request_id);
}

- (void)runBackgroundLoop {
  run_background_loop_func_();
}

- (void)runBackgroundStreamBackLoop {
  run_background_stream_back_loop_func_();
}

- (void)exitBackgroundLoop {
  exit_background_loop_func_();
}

@end
