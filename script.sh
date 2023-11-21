export DIR=/data1/ssq/experiment/RSpell
export MODEL_NAME=glyce
export FONT_TYPE=sim
export CHECKPOINT=Results/pre-157500
export CUDA_VISIBLE_DEVICES=0,4
python -m torch.distributed.launch   --master_port 29506\
    --nproc_per_node 2 $DIR/Code/train_baseline_js.py \
	--model_name $DIR/Transformers/${MODEL_NAME} \
	--train_files $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law.train\
	--train_files1 $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law_js.train\
	--train_files2 $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law_js.trainsep\
	--train_files3 $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law.traintgt\
	--val_files $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law.dev\
	--val_files1 $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law_js.dev\
	--val_files2 $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law_js.devsep\
	--val_files3 $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law.devtgt\
	--test_files $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/domain/law_js.test \
	--cached_dir $DIR/Cache \
	--result_dir $DIR/Results/pre157500_lawjs \
	--glyce_config_path $DIR/Transformers/glyce_bert_both_font.json \
	--vocab_file $DIR/Data/vocab/allNoun.txt \
	--load_pretrain_checkpoint ${CHECKPOINT} \
	--checkpoint_index 157500 \
	--font_type ${FONT_TYPE} \
	--overwrite_cached True \
	--num_train_epochs 200 \
	--gradient_accumulation_steps 2 \
	--use_pinyin True \
	--use_word_feature False \
	--use_copy_label False \
	--compute_metrics True \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--save_steps 50 \
	--logging_steps 500 \
	--fp16 True \
	--do_test False \


# nohup bash script.sh > pre157500_lawjs.log 2>&1 &






