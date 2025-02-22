import requests
from ..util import is_model
from ..util.env import HF_USER, W_TOKEN, R_TOKEN
from huggingface_hub import upload_file, create_repo, file_exists, repo_exists, upload_folder
import os

def determine_repo_id(model_name:str):
    is_existing_repo = repo_exists(f"{HF_USER}/{model_name}", token=R_TOKEN)
    ends_with_version_number = model_name[-2].startswith('-v')
    if is_existing_repo and ends_with_version_number:
        version_number = int(model_name[-1])
        new_version_repo_name =  f"{model_name[:-2]}-v{version_number+1}"
        return determine_repo_id(new_version_repo_name)
    elif is_existing_repo:
        name_with_version = f"{model_name}-v2"
        return determine_repo_id(name_with_version)
    else:
        return f"{HF_USER}/{model_name}"

def upload_models_to_hf(model_name:str, upload_dir:str):
    repo_name = determine_repo_id(model_name)
    repo_exists(repo_name, token=R_TOKEN)
    create_repo(repo_name, token=W_TOKEN, private=True, exist_ok=True)
    file_list = list(os.listdir(upload_dir))
    models = list(filter(lambda file: is_model(file) == True, file_list))
    
    for file_path in models:
        base_name = os.path.basename(file_path)
        file_full_path=f"{upload_dir}/{file_path}"
        if not file_exists(model_name, base_name):
            upload_file(path_or_fileobj=file_full_path, path_in_repo=base_name, repo_id=repo_name, token=W_TOKEN)
        else:
            print(f"file existing: {base_name}")
    print("model uploaded!")

def upload_all_files_to_hf(repo_id:str, local_dir:str):
    try:
        create_repo(repo_id,token=W_TOKEN, private=True, exist_ok=True)
        upload_folder(folder_path=local_dir, repo_id=repo_id, token=W_TOKEN)
    except:
        print("upload to hf failed!")
        raise

def call_uploader(upload_dir:str):
    print("calling uploader")
    uploader_url = os.environ.get("UPLOADER_URL", "")
    working_repo = os.environ.get("WORKING_REPO", "")
    repo_dir = os.environ.get("REPO_DIR", "")
    write_token = os.environ.get("W_TOKEN", "")
    payload  = {
        "input":{
            "upload_dir": upload_dir,
            "repo_id": working_repo,
            "repo_path": repo_dir,
            "write_token": write_token
        }
    }
    
    print(f"will be called with these payload: {repr(payload)}")
    uploader_key = os.environ.get("UPLOADER_KEY", "")
    requests.post(uploader_url, json=payload, headers={"Authorization": f"Bearer {uploader_key}"})

def upload_file_future(file_path:str, path_in_repo:str):
    print("uploading file to hf with future")
    working_repo = os.environ.get("WORKING_REPO", "")
    write_token = os.environ.get("W_TOKEN", "")
    upload_file(repo_id=working_repo, path_or_fileobj=file_path, path_in_repo=path_in_repo, token=write_token, run_as_future=True)
