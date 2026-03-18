WORK_DIR=$(pwd)
LUSTRE_DIR=/lustre
ACCOUNT=llmservice_modelalignment_sft
#export VLLM_CACHE_DIR=/lustre/fs1/portfolios/llmservice/users/adithyare/vllm_cache

CONTAINER_IMAGE="/lustre/fs2/portfolios/llmservice/users/lvega/sqsh/vllm-openai-v0.10.0.sqsh"
CONTAINER_IMAGE="/lustre/fs2/portfolios/llmservice/users/lvega/sqsh/vllm-logits-processor-09-09-25.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/adithyare/containers/vllm_budget_091825.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/dmosallanezh/containers/vllm-nvidia-nano-next-09-17-25.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/adithyare/container_sqsh_files/nemotron5-111524_mamba_ssm_222_fix.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/adithyare/code/budget_vs_milestones/openmathreasoning/nemo-rl-nano-v3-rc6-prebuilt.squashfs"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/adithyare/code/budget_vs_milestones/openmathreasoning/nemo-rl-nano-v3-rc6-prebuilt-vllm.squashfs"

srun -A ${ACCOUNT} \
	-p interactive \
	--time=04:00:00 \
	--no-container-mount-home \
	--container-image=${CONTAINER_IMAGE} \
	--container-mounts=${WORK_DIR}:/workdir,${LUSTRE_DIR}:/lustre \
	--gres=gpu:1 \
	--pty bash -c 'cd /workdir; exec bash'
