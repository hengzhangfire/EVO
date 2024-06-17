import os
import torch


class zz(object):
    def __init__(self, model_type, model_name_or_path, task_name, do_train, do_eval, do_lower_case, data_dir, output_dir,randomSeed):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_lower_case = do_lower_case
        self.data_dir = data_dir
        #self.max_seq_length = max_seq_length
        #self.per_gpu_train_batch_size = per_gpu_train_batch_size
        #self.learning_rate = learning_rate 见下
        self.num_train_epochs = 1.0
        self.output_dir = output_dir
        # 默认赋值
        self.config_name = ''
        self.tokenizer_name = ''
        self.cache_dir = ''
        self.evaluate_during_training = False
        self.per_gpu_eval_batch_size = 32 #checked "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
        # self.gradient_accumulation_steps = 1 #见下
        # self.weight_decay = 0.0
        self.adam_epsilon = 1e-8 #parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        self.max_grad_norm = 1.0 #parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.max_steps = -1#-1 #见下
        self.warmup_steps = 0#0
        self.logging_steps = -1#50 #parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
        self.save_steps = -1# #parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True# lyx 运行时需要重写
        self.overwrite_cache = False
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''
        self.per_gpu_train_batch_size = 32 # 16太大了用不了，改成8
        self.seed = randomSeed#baseline
        self.weight_decay = 1e-2

        self.learning_rate = 1e-5
        self.gradient_accumulation_steps = 1  # 只有QNLI是4
        self.max_seq_length = 32
        #self.max_steps = 123873
        #self.warmup_steps = 7432

        # QNLI
        '''self.learning_rate = 2e-5
        self.gradient_accumulation_steps = 1  # 只有QNLI是4
        self.max_seq_length = 32
        #self.max_steps = 33112
        #self.warmup_steps = 1986'''
        # SST-2
        '''self.learning_rate = 1e-5
        self.per_gpu_train_batch_size = 32  # 不修改 我的gpu好像最多到8
        self.gradient_accumulation_steps = 1  # 只有QNLI是4
        self.max_seq_length = 32
        self.max_steps = 20935
        self.warmup_steps = 1256'''
