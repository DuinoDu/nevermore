from pint_horizon.aidi import traincli
import os
cfg = dict(
    gpus="0,1,2,3",
    num_machines=1,
    submit_root="./",
    job_name="NYUv2-multitask03333-lr2e-5-bs16-4gpus-precision16",
    job_pwd=5160,
    project_id="TD2021002",
    docker_image="docker.hobot.cc/imagesys/hdlt:fsd_multitask-cu10-20210621-v0.3",  # noqa
    # docker_image="docker.hobot.cc/imagesys/hdlt:fsd_multitask-cu11-202106121-v0.4",  # noqa
    job_list=[
        "pip3 install pytorch-lightning --user",
        "pip3 install -r requirements.txt",
        "pip3 install torchmetrics --user",
        # "pip3 install hydra-core --upgrade --user -i https://pypi.tuna.tsinghua.edu.cn/simple/",
        "mkdir -p /home/users/dixiao.wei/.cache/torch/hub/checkpoints",
        "cp /cluster_home/custom_data/vgg16-397923af.pth /home/users/dixiao.wei/.cache/torch/hub/checkpoints/",  # noqa
        # "sleep 1000m",
        "make train",
    ],
    task_label="mnist",
    priority=5,
    no_softlink=True
)
 
# data_root = 'data'
# if os.path.exists(data_root):
#     os.system(f'mv {data_root} /tmp/')

traincli(cfg)

# if os.path.exists(f'/tmp/{data_root}'):
#     os.system(f'mv /tmp/{data_root} ./')