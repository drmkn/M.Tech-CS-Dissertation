# PREFIX = '/home/saptarshi/Dhruv/'
PREFIX = '/user1/student/mtc/mtc2023/cs2306/Dhruv/Code/'
CONFIG = {'syn' : {'seed' : 10,'train_samples' : 2000, 'test_samples' : 600,
                   'name' : 'syn', 'train_data' : PREFIX + 'Dissertation/datasets/synthetic_dataset/syn-train.csv',
                   'test_data' : PREFIX + 'Dissertation/datasets/synthetic_dataset/syn-test.csv',
                   'dag_path' : PREFIX + 'Dissertation/dags_estimated/syn_dag.json',
                   'graph_path' : PREFIX +'Dissertation/assets/est_syn_dag.png',
                   'to_drop' : ['Y'], 'var_names' : ['W','Z','X'], 'w_threshold' : 0.1,
                   'lambda2' : 0.001, 'lambda1' : 0.01
                   },

          'mpg' : {'seed' : 10,'train_samples' : 274, 'test_samples' : 118,
                   'name' : 'syn', 'train_data' : PREFIX + 'Dissertation/datasets/auto+mpg/mpg-train.csv',
                   'test_data' : PREFIX + 'Dissertation/datasets/auto+mpg/mpg-test.csv',
                   'dag_path' : PREFIX + 'Dissertation/dags_estimated/mpg_dag.json',
                   'graph_path' : PREFIX + 'Dissertation/assets/est_mpg_dag.png',
                   'to_drop' : ['M'], 'var_names' : ['C','D','H','W','A'], 'w_threshold' : 0.3,
                   'lambda2' : 0.01, 'lambda1' : 0.1
                   }        
                   }