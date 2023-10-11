from autograd_lib import autograd_lib
import torch
import numpy as np

# The gradient norm metric
class grad_norm:
    def __init__(self, X, y):
        
        self.X=X
        self.y=y
        self._grads = None

    def gradient(self, net, device):
            """
            Used to compute the norm of the gradient through training
            Args:
              net: pytorch neural network
              device: device to run the computation on
            """

            # setup
            data = torch.tensor(self.X, device=device)
            targets = torch.tensor(self.y, device=device).long()
            loss_fn = torch.nn.L1Loss()

            model = net

            # register the model for autograd
            autograd_lib.register(model)

            activations = {}

            def save_activations(layer, A, _):
                activations[layer] = A

            with autograd_lib.module_hook(save_activations):
                output = model(data)
                loss = loss_fn(output, targets)

            norms = [torch.zeros(data.shape[0], device=device)]

            def per_example_norms(layer, _, B):
                A = activations[layer]
                norms[0] += (A * A).sum(dim=1) * (B * B).sum(dim=1)

            with autograd_lib.module_hook(per_example_norms):
                loss.backward()

            grads_train = norms[0].cpu().numpy()

            if self._grads is None:  # Happens only on first iteration
                self._grads = np.expand_dims(grads_train, axis=-1)
            else:
                stack = [self._grads, np.expand_dims(grads_train, axis=-1)]
                self._grads = np.hstack(stack)


# variance of gradients metric
def vog_scores(grad_X, grad_Y, vog):

    mode='complete'

    # Analysis of Variance of Gradients
    training_vog_stats=[]
    training_labels=[]
    # training_class_variances = list(list() for i in range(2))
    # training_class_variances_stats = list(list() for i in range(2))
    for ii in range(grad_X.shape[0]):
        if mode == 'early':
            temp_grad = np.array(vog[ii][:5])
        elif mode == 'middle':
            temp_grad = np.array(vog[ii][5:10])
        elif mode == 'late':
            temp_grad = np.array(vog[ii][10:])
        elif mode == 'complete':
            temp_grad = np.array(vog[ii])   

        mean_grad = np.sum(np.array(vog[ii]), axis=0)/len(temp_grad)
        training_vog_stats.append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
        training_labels.append(int(grad_Y[ii].item()))
    
    return training_vog_stats, training_labels
