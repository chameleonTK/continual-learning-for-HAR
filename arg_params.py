import argparse
from generator_multiple_gan import GeneratorMultipleGAN
from generator_single_gan import GeneratorSingleGAN

class IterAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(IterAction, self).__init__(option_strings, dest, **kwargs)


    def __call__(self, parser, namespace, values, option_string=None):
        if option_string=="--iters":
            setattr(namespace, "iters", int(values))
            setattr(namespace, "log", int(values*0.05))
        else:
            setattr(namespace, "g_iters", int(values))
            setattr(namespace, "g_log", int(values*0.05))

def get_parser():
    parser = argparse.ArgumentParser('./main.py', description='Run continual learning experiment.')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--method', type=str, choices=['generative', 'exact', 'none', "offline"], dest='replay', default='adam', help="method")
    parser.add_argument('--generative-model', type=str, help="path to trained generative model")
    parser.add_argument('--output-model-path', type=str,help="path for output")
    parser.add_argument('--results-dir', required=True, type=str,help="path for results")
    parser.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
    parser.add_argument('--log', type=int, default=200, help="# iters after which to plot solver loss")
    parser.add_argument('--g-log', type=int, default=200, help="# iters after which to plot generator loss")

    task_params = parser.add_argument_group('Task Parameters')
    task_params.add_argument('--data-dir', type=str, dest='data_dir', required=True, help="default: %(default)s")
    task_params.add_argument('--tasks', type=int, default=5, help='number of tasks')

    train_params = parser.add_argument_group('Training Parameters')
    train_params.add_argument('--iters', type=int, default=1000, action=IterAction, help="# iterator to optimize solver")
    train_params.add_argument('--lr', type=float, default=0.001, help="learning rate for solver")

    train_params.add_argument('--g-iters', type=int, default=5000, action=IterAction, help="# iterator to optimize generator")
    train_params.add_argument('--lr-gen', type=float, default=0.001, help="learning rate for generator")

    train_params.add_argument('--batch', type=int, default=5, help="batch-size")
    train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')
    train_params.add_argument('--replay-size', type=int, default=500, help="# replayed samples used in each training session")
    train_params.add_argument('--rnt', type=float, default=0.5, help="relative importance of new task")


    model_params = parser.add_argument_group('Model Parameters')
    model_params.add_argument('--solver-fc-layers', type=int, default=3, help="# of fully-connected layers")
    model_params.add_argument('--solver-fc-units', type=int, default=500, help="# of units in first fc-layers")

    generator_params = parser.add_argument_group('Generator Parameters')
    generator_params.add_argument('--critic-fc-layers', type=int, default=3, help="[critic] # of fully-connected layers")
    generator_params.add_argument('--critic-fc-units', type=int, default=100, help="[critic] # of units in first fc-layers")
    generator_params.add_argument('--critic-lr', type=float, default=0.001, help="[critic] learning rate")
    generator_params.add_argument('--generator-fc-layers', type=int, default=3, help="[generator] # of fully-connected layers")
    generator_params.add_argument('--generator-fc-units', type=int, default=100, help="[generator] # of units in first fc-layers")
    generator_params.add_argument('--generator-lr', type=float, default=0.001, help="[generator] learning rate")

    generator_params.add_argument('--generator-z-size', type=int, default=20, help='[generator] size of latent representation')
    generator_params.add_argument('--generator-activation', type=str, choices=['sigmoid', 'relu', 'identity'], default='relu', help='[generator] Output function')


    component_params = parser.add_argument_group('Additional Component Parameters')
    parser.add_argument('--self-verify', action='store_true', help="enable self-verifing")
    parser.add_argument('--oversampling', action='store_true', help="enable oversampling")
    parser.add_argument('--solver-ewc', action='store_false', help="enable EWC regularisation")
    parser.add_argument('--solver-distill', action='store_true', help="enable knowledge distilling")
    parser.add_argument('--generator-noise', action='store_true', help="enable instance noise")

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    # args.data_dir = "../../Data/twor.2009/annotated.feat.ch15"

    
    
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

    