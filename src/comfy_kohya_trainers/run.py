import os
import subprocess
parts = os.path.dirname(os.path.abspath(__file__))
to_add = "/".join(parts.split("/")[:-2])
# print(to_add)
python_path = subprocess.check_output("which python", shell=True, text=True).strip()
print(python_path)
# print(python_path)

cmd = f"{python_path} -m src.my_custom_nodepack.arg_test --mixed_precision fp16 --save_precision fp16 --xformers"
# # cmd = "echo 1"
env={
    "PYTHONPATH": to_add
}
subprocess.call(cmd, shell=True, env=env)
