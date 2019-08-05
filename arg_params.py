import argparse
from generator_multiple_gan import GeneratorMultipleGAN
from generator_single_gan import GeneratorSingleGAN

def get_parser():
    parser = argparse.ArgumentParser('./main.py', description='Run continual learning experiment.')
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    args.seed = 10
    args.tasks = 5
    args.data_dir = "../../Data/twor.2009/annotated.feat.ch15"
    args.result_dir = "./results"
    args.plot_dir = "./plots"

    args.iters = 1000                # batches to optimize solver
    args.lr = 0.001                 # learning rate
    args.batch = 5                  # batch-size
    args.optimizer = "adam"         # ['adam', 'adam_reset', 'sgd']

    args.solver_fc_layers = 3
    args.solver_fc_units = 500

    args.critic_fc_layers = 3
    args.critic_fc_units = 100
    args.critic_lr = 0.001
    args.generator_fc_layers = 3
    args.generator_fc_units = 100
    args.generator_lr = 0.001
    args.generator_z_size = 20
    args.generator_activation = "relu"

    args.replay = "generative"
    args.generative_model = None #"sg-cgan-iter5000.model"
    args.output_model_path = None
    args.lr_gen = 0.001             # learning rate for generator
    args.g_iters = 5000              # batches to optimize solver
    args.visdom = True
    args.log = 200
    args.g_log = int(args.g_iters*0.05)

    args.self_verify = True
    args.oversampling = True
    args.solver_ewc = False
    args.solver_distill = False
    args.generator_noise = False


    # args.pre_training = False
    # args.evaluate = False
    
    return args

def get_generator(model, config, cuda, device, args, init_n_classes=2):
    if model == "mp-gan":
        return GeneratorMultipleGAN(
            input_feat=config['feature'],
            model="gan",
            cuda=cuda,
            device=device,

            z_size=args.generator_z_size,
            critic_fc_layers=args.critic_fc_layers, critic_fc_units=args.critic_fc_units, critic_lr=args.critic_lr, 
            generator_fc_layers=args.generator_fc_layers, generator_fc_units=args.generator_fc_units, generator_lr=args.generator_lr, 
            generator_activation=args.generator_activation,

            critic_updates_per_generator_update=5,
            gp_lamda=10.0
        )
    elif model == "mp-wgan":
        return GeneratorMultipleGAN(
            input_feat=config['feature'],
            model="wgan",
            cuda=cuda,
            device=device,

            z_size=args.generator_z_size,
            critic_fc_layers=args.critic_fc_layers, critic_fc_units=args.critic_fc_units, critic_lr=args.critic_lr, 
            generator_fc_layers=args.generator_fc_layers, generator_fc_units=args.generator_fc_units, generator_lr=args.generator_lr, 
            generator_activation=args.generator_activation,

            critic_updates_per_generator_update=5,
            gp_lamda=10.0
        )
    elif model == "sg-cgan":
        return GeneratorSingleGAN(
            input_feat=config['feature'],
            classes=init_n_classes,
            model="cgan",
            cuda=cuda,
            device=device,

            z_size=args.generator_z_size,
            critic_fc_layers=args.critic_fc_layers, critic_fc_units=args.critic_fc_units, critic_lr=args.critic_lr, 
            generator_fc_layers=args.generator_fc_layers, generator_fc_units=args.generator_fc_units, generator_lr=args.generator_lr,
            generator_activation=args.generator_activation,
        )
    elif model == "sg-cwgan":
        return GeneratorSingleGAN(
            input_feat=config['feature'],
            classes=init_n_classes,
            model="cwgan",
            cuda=cuda,
            device=device,

            z_size=args.generator_z_size,
            critic_fc_layers=args.critic_fc_layers, critic_fc_units=args.critic_fc_units, critic_lr=args.critic_lr, 
            generator_fc_layers=args.generator_fc_layers, generator_fc_units=args.generator_fc_units, generator_lr=args.generator_lr,
            generator_activation=args.generator_activation,
        )
    else:
        raise Exception("Unknown model")

    