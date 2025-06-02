from transformers import modeling_flash_attention_utils 

def wrap(FA_fn):
    def wraped_FA_fn(*args, **kwargs):
        assert kwargs.pop("dropout_p", 0.0) == 0.0
        assert kwargs.pop("softcap", 0.0) == 0.0
        return FA_fn(*args, **kwargs)[0]
    return wraped_FA_fn

def patch_FA3():
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    modeling_flash_attention_utils.flash_attn_func = wrap(flash_attn_func)
    modeling_flash_attention_utils.flash_attn_varlen_func = wrap(flash_attn_varlen_func)