def get_optimizer_args(args_dict: dict) -> dict:
    keys_that_has_values = [k for k, v in args_dict.items() if v is not None or v != ""]
    if len(keys_that_has_values) == 0:
        return {}
    else:
        joined = [f"{k}={v}" for k, v in args_dict.items() if k in keys_that_has_values]
        return {
            "optimizer_args": joined,
        }
