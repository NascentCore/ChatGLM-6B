# Dockerfile

运行下面命令构建 docker 镜像
```shell
# 编译镜像
docker build -f dockerfile_base -t chatglm_base .

# 进入容器内部
# docker exec -it 10582be21396 /bin/bash # xxxxxxx 是PS后容器的CONTAINER ID号
docker run --name=glm --runtime=nvidia -itd -p 2222:22 chatglm_base
```
