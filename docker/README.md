# Docker

用于记录 Docker 镜像相关的代码

## 构建 Docker 镜像

运行下面命令构建 docker 镜像
```shell
# 编译镜像
docker build -f dockerfile_base -t chatglm_base .

# 进入容器内部
# docker exec -it 10582be21396 /bin/bash # xxxxxxx 是PS后容器的CONTAINER ID号
docker run --name=glm --runtime=nvidia -itd -p 2222:22 chatglm_base
```

## 错误记录

result_2e-2_128_2023-05-23-11.out 是运行上面的步骤生成的 docker 镜像的日志
日志中记录了错误信息
