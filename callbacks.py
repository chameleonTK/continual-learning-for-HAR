import visual_visdom


def _solver_loss_cb(log, visdom, model=None, tasks=None, iters_per_task=None, replay=False, progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    def cb(bar, iter, loss_dict, task=1):
        '''Callback-function, to call on every iteration to keep track of training progress.'''
        if task is None:
            task = 0
            
        # iteration = (task-1)*iters_per_task + iter
        # progress-bar
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            bar.set_description(
                '  <SOLVER>   |{t_stm} training loss: {loss:.3} | training precision: {prec:.3} |'
                    .format(t_stm=task_stm, loss=loss_dict['loss_total'], prec=loss_dict['precision'])
            )
            bar.update(1)

        # log the loss of the solver (to visdom)
        # if (iteration % log == 0) and (visdom is not None):
        #     plot_data = [loss_dict['loss_total']]
        #     names = ["accuracy"]

            # visual_visdom.visualize_scalars(
            #     scalars=plot_data, names=names, iteration=iteration,
            #     title="Solver loss", env=visdom["env"], ylabel="training loss"
            # )

    # Return the callback-function.
    return cb


def _task_loss_cb(model, test_datasets, log, visdom, iters_per_task, vis_name=""):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    def cb(iter, task=1):
        '''Callback-function, to call on every iteration to keep track of training progress.'''
        if task is None:
            task = 0
            
        iteration = (task-1)*iters_per_task + iter
        if (iteration % log == 0) and (visdom is not None):
            loss_dict = model.test(task, test_datasets, verbose=False)
            while len(loss_dict["Precision"]) < len(test_datasets):
                loss_dict["Precision"].append(0)
            
            loss_dict["Task"] = range(len(test_datasets))
            plot_data = loss_dict["Precision"]
            
            visdom["values"].append({"iter": iteration, "acc": plot_data})
            names = ["task"+str(s+1) for s in loss_dict["Task"]]

            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="Task accuracy"+vis_name, env=visdom["env"], ylabel="accuracy per task"
            )

    # Return the callback-function.
    return cb



def _generator_training_callback(log, visdom, model, tasks=None, iters_per_task=None, replay=False, progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    def cb(bar, iter, loss_dict, task=1):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        iteration = iter

        # progress-bar
        if progress_bar and bar is not None:
            task_stm = " Class: {} |".format(task)
            bar.set_description(
                '  <GAN>      |{t_stm} d cost: {loss:.3} | g cost: {prec:.3} |'
                    .format(t_stm=task_stm, loss=loss_dict['d_cost'], prec=loss_dict['g_cost'])
            )
            bar.update(1)

        # if (iteration % log == 0) and (visdom is not None):

        #     plot_data = [loss_dict['d_cost'], loss_dict['g_cost']]
        #     names = ['Discriminator cost', 'Generator cost']

        #     visual_visdom.visualize_scalars(
        #         scalars=plot_data, names=names, iteration=iteration,
        #         title="GENERATOR: loss class{t}".format(t=task), env=visdom["env"], ylabel="training loss"
        #     )
    # Return the callback-function
    return cb
