# Usage:
# make install			# downloads miniconda and initializes conda environment
# make install_mimic	# downloads required mimic files from physionet (physionet credentialed account required)
# make server	  		# starts mlflow server at port 5000
# make run  			# executes main.py within the conda environment \
				  			example: make run ARGS="--experimentconfig_sequence_type huawei_logs"
# make run_mimic 		# executes main.py within the conda environment for all knowledge types on mimic dataset
# make run_huawei		# executes main.py within the conda environment for all knowledge types on huawei dataset

CONDA_ENV_NAME = aiops
CONDA_URL = https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
CONDA_SH = Miniconda3-py38_4.12.0-Linux-x86_64.sh
CONDA_DIR = .tmp

DATA_DIR = data
KNOWLEDGE_TYPES = simple gram text causal_heuristic causal_score causal_constraint
BGL_KNOWLEDGE_TYPES = simple text causal_heuristic causal_score causal_Fast-IAMB-smc-cor

install:
ifneq (,$(wildcard ${CONDA_DIR}))
	@echo "Remove old install files"
	@rm -Rf ${CONDA_DIR}
endif
	@echo "Downloading miniconda..."
	@mkdir ${CONDA_DIR}
	@cd .tmp && wget -nc ${CONDA_URL} > /dev/null
	@chmod +x ./${CONDA_DIR}/${CONDA_SH}
	@./${CONDA_DIR}/${CONDA_SH} -b -u -p ./${CONDA_DIR}/miniconda3/ > /dev/null
	@echo "Initializing conda environment..."
	@./${CONDA_DIR}/miniconda3/bin/conda env create --name ${CONDA_ENV_NAME} --file environment.yml > /dev/null
	@echo "Finished!"

install_mimic:
	@cd ${DATA_DIR}
	@wget -N -c --user ${PHYSIONET_USER} --ask-password https://physionet.org/files/mimiciii/1.4/ADMISSIONS.csv.gz
	@wget -N -c --user ${PHYSIONET_USER} --ask-password https://physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz
	@gzip -d ADMISSIONS.csv.gz
	@gzip -d DIAGNOSES_ICD.csv.gz

server:
	@echo "Starting MLFlow UI at port 5000"
	PATH="${PATH}:$(shell pwd)/${CONDA_DIR}/miniconda3/bin" ; \
	./${CONDA_DIR}/miniconda3/bin/mlflow server --gunicorn-opts -t180 --host 0.0.0.0 --port 5000

notebook:
	@echo "Starting Jupyter Notebook at port 8888"
	PATH="${PATH}:$(shell pwd)/${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin" ; \
	./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/jupyter notebook notebooks/ --no-browser 

run: 
	./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py ${ARGS}

run_mimic: 
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		echo "Starting experiment for mimic with knowledge type " $$knowledge_type "....." ; \
		./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_sequence_type mimic \
			--experimentconfig_model_type $$knowledge_type \
			--mimicpreprocessorconfig_sequence_column_name level_all \
		    --mimicpreprocessorconfig_prediction_column level_0 \
			--sequenceconfig_x_sequence_column_name level_0 \
			--sequenceconfig_y_sequence_column_name level_3 \
			--sequenceconfig_predict_full_y_sequence_wide \
			${ARGS} ; \
	done ; \

run_huawei:
	echo "Starting experiment for huawei_logs with $(type) knowledge ....." ; \
	./${CONDA_DIR}/miniconda3/bin/python3.8 main.py \
		--experimentconfig_sequence_type huawei_logs \
		--experimentconfig_model_type $(type) \
		--experimentconfig_multilabel_classification \
		--experimentconfig_batch_size 128 \
		--experimentconfig_n_epochs 100 \
		--experimentconfig_load_knowledge_df \
		--experimentconfig_serialize_knowledge_df \
		--experimentconfig_knowledge_df_file data/final/full/huawei/without_ts/$(type)_knowledge_df.csv \
		--huaweipreprocessorconfig_min_causality 0.01 \
		--huaweipreprocessorconfig_aggregated_log_file data/final/$(size)/huawei/Huawei_$(size).csv \
		--huaweipreprocessorconfig_fine_drain_log_depth 10 \
		--huaweipreprocessorconfig_fine_drain_log_st 0.75 \
		--sequenceconfig_x_sequence_column_name fine_log_cluster_template \
		--sequenceconfig_y_sequence_column_name attributes \
		--sequenceconfig_max_window_size 10 \
		--sequenceconfig_min_window_size 10 \
		--sequenceconfig_flatten_y \
		--modelconfig_rnn_type gru \
		--modelconfig_rnn_dim 200 \
		--modelconfig_embedding_dim 300 \
		--modelconfig_attention_dim 100 \
		--timeseriestransformerconfig_bin_overlap 00:00:01 \
		--timeseriestransformerconfig_bin_size 00:00:05 \
		--no-modelconfig_base_hidden_embeddings_trainable \
		${ARGS} ; \

run_huawei_with_ts:
	echo "Starting experiment for huawei_logs with $(type) knowledge ....." ; \
	./${CONDA_DIR}/miniconda3/bin/python3.8 main.py \
		--experimentconfig_sequence_type huawei_logs \
		--experimentconfig_model_type simple \
		--experimentconfig_multilabel_classification \
		--experimentconfig_batch_size 128 \
		--experimentconfig_n_epochs 100 \
		--experimentconfig_load_knowledge_df \
		--experimentconfig_serialize_knowledge_df \
		--experimentconfig_knowledge_df_file data/final/full/huawei/with_ts/$(type)_knowledge_df.csv \
		--huaweipreprocessorconfig_min_causality 0.01 \
		--huaweipreprocessorconfig_aggregated_log_file data/final/$(size)/huawei/Huawei_$(size).csv \
		--huaweipreprocessorconfig_fine_drain_log_depth 10 \
		--huaweipreprocessorconfig_fine_drain_log_st 0.75 \
		--sequenceconfig_x_sequence_column_name fine_log_cluster_template \
		--sequenceconfig_y_sequence_column_name attributes \
		--sequenceconfig_max_window_size 10 \
		--sequenceconfig_min_window_size 10 \
		--sequenceconfig_flatten_y \
		--modelconfig_rnn_type gru \
		--modelconfig_rnn_dim 200 \
		--modelconfig_embedding_dim 300 \
		--modelconfig_attention_dim 100 \
		--timeseriestransformerconfig_bin_overlap 00:00:01 \
		--timeseriestransformerconfig_bin_size 00:00:05 \
		--no-modelconfig_base_hidden_embeddings_trainable \
		--no-huaweipreprocessorconfig_remove_dates_from_payload \
		${ARGS} ; \

run_hdfs:
	echo "Starting experiment for HDFS logs with $(type) knowledge ....." ; \
	./${CONDA_DIR}/miniconda3/bin/python3.8 main.py \
		--experimentconfig_sequence_type hdfs_logs \
		--experimentconfig_model_type $(type) \
		--experimentconfig_multilabel_classification \
		--experimentconfig_batch_size 32 \
		--experimentconfig_n_epochs 100 \
		--experimentconfig_load_knowledge_df \
		--experimentconfig_only_generate_knowledge \
		--experimentconfig_knowledge_df_file data/final/full/hdfs/knowledge/$(type)_knowledge_df.csv \
		--hdfspreprocessorconfig_aggregated_log_file data/final/$(size)/hdfs/HDFS_$(size).csv \
		--hdfspreprocessorconfig_fine_drain_log_depth 10 \
		--hdfspreprocessorconfig_fine_drain_log_st 0.75 \
		--sequenceconfig_x_sequence_column_name fine_log_cluster_template \
		--sequenceconfig_y_sequence_column_name attributes \
		--sequenceconfig_max_window_size 10 \
		--sequenceconfig_min_window_size 10 \
		--sequenceconfig_flatten_y \
		--modelconfig_rnn_type gru \
		--modelconfig_rnn_dim 200 \
		--modelconfig_embedding_dim 300 \
		--modelconfig_attention_dim 100 \
		--no-modelconfig_base_hidden_embeddings_trainable \
		--timeseriestransformerconfig_bin_size 00:00:05 \
		--timeseriestransformerconfig_bin_overlap 00:00:01 \
		${ARGS} ; \

run_bgl:
	echo "Starting experiment for BGL logs with $(type) knowledge ....." ; \
	./${CONDA_DIR}/miniconda3/bin/python3.8 main.py \
		--experimentconfig_sequence_type bgl_logs \
		--experimentconfig_model_type $(type) \
		--experimentconfig_multilabel_classification \
		--experimentconfig_batch_size 32 \
		--experimentconfig_n_epochs 100 \
		--experimentconfig_load_knowledge_df \
		--experimentconfig_only_generate_knowledge \
		--experimentconfig_knowledge_df_file data/final/full/bgl/knowledge/$(type)_knowledge_df.csv \
		--bglpreprocessorconfig_aggregated_log_file data/final/$(size)/bgl/BGL_$(size).csv \
		--bglpreprocessorconfig_fine_drain_log_depth 10 \
		--bglpreprocessorconfig_fine_drain_log_st 0.75 \
		--sequenceconfig_x_sequence_column_name fine_log_cluster_template \
		--sequenceconfig_y_sequence_column_name attributes \
		--sequenceconfig_max_window_size 10 \
		--sequenceconfig_min_window_size 10 \
		--sequenceconfig_flatten_y \
		--modelconfig_rnn_type gru \
		--modelconfig_rnn_dim 200 \
		--modelconfig_embedding_dim 300 \
		--modelconfig_attention_dim 100 \
		--no-modelconfig_base_hidden_embeddings_trainable \
		--timeseriestransformerconfig_bin_size 00:00:05 \
		--timeseriestransformerconfig_bin_overlap 00:00:01 \
		${ARGS} ; \