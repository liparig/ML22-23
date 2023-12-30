import json
import costants as C

# it's a nice to have but it's not completed



theta_batch = {C.L_NET:[[17,4,1]],
            C.L_ACTIVATION:[[C.TANH]],
            C.L_ETA:[0.8 ,0.5],
            C.L_TAU: [(500,0.08)],
            C.L_REG:[(False,False)],
            C.L_DIMBATCH:[0],
            C.L_MOMENTUM: [(False,False)],
            C.L_EPOCHS:[2000],
            C.L_SHUFFLE:True,
            C.L_EPS: [0.1 , 0.3],
            C.L_DISTRIBUTION:[C.UNIFORM],
            C.L_BIAS:[0],
            C.L_SEED: [52],
            C.L_CLASSIFICATION:True,
            C.L_EARLYSTOP:True,
            C.L_PATIENCE: [30],
            C.L_TRESHOLD_VARIANCE:[C.TRESHOLDVARIANCE]
        
    }
theta_mini = theta_batch.copy()
theta_mini[C.L_ETA]=[0.05]
theta_mini[C.L_TAU]=[(30, 0.005),(70 , 0.005)]
theta_mini[C.L_DIMBATCH]=[20]
theta_mini[C.L_MOMENTUM]= [(C.NESTEROV,0.9)]
thetas={"batch_monk1":theta_batch,"mini_batch":theta_mini}
with open('Project_Hyperparameters', 'w',encoding='utf-8') as json_file:     json.dump(thetas, json_file, ensure_ascii=False, indent=2)