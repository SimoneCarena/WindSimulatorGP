import gpytorch
import torch
import matplotlib.pyplot as plt

def train_ExactGP__(train_data, train_labels, models, likelihoods, name, training_iter):
    '''
    Trains the exact GP model given the specified kernel.\\
    The trained model is saved in the `models/ExactGP` folder
    '''
    axis = ['x','y']
    l = [(models[i], likelihoods[i], train_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
        # Setup each model
        model.train()
        likelihood.train()
        optimizer=torch.optim.Adam(model.parameters(), lr=0.1)
        mll=gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Train each model
        print('Training ExactGP model ({}-axis) on {} iterations using {} kernel'.format(axis,training_iter,name))
        for i in range(training_iter):
            # Zero gradients from the previous iteration
            optimizer.zero_grad()
            # Output from the model
            output = model(train_data)
            # Compute the loss and backpropagate the gradients
            # We want to maximize the log-likelihood on the parameters, so
            # we minimize -mll
            loss = -mll(output, labels)
            loss.backward()
            optimizer.step()
            print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
        print('')

        # Save the model
        torch.save(model.state_dict(),f'models/ExactGP/model-{axis}-{name}.pth')
        torch.save(likelihood.state_dict(),f'models/ExactGP/likelihood-{axis}-{name}.pth')

    print(f'Training of ExactGP model with {name} kernel complete!\n')


def test_ExactGP__(test_data, test_labels, models, likelihoods, name, T, save=False, show=True):
    '''
    Test a trained GP model.\\
    '''
    axis = ['x','y']
    l = [(models[i], likelihoods[i], test_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            observed_pred = model(test_data)
        lower, upper = observed_pred.confidence_region()
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.suptitle(f'ExactGP Wind Estimation ({axis}-axis) with {name} kernel')
        ax.plot(T,labels,color='orange',label='True Data')
        ax.plot(T,observed_pred.mean.numpy(),'b-',label="Estimated Data")
        ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='cyan',label='Confidence')
        ax.legend()
        if save:
            plt.savefig(f'imgs/gp_plots/ExactGP/{name}-{axis}',dpi=300)

    if show:
        plt.show()
    plt.close()


def train_ExactMultiOutputExactGP__(train_data, train_labels, model, likelihood, name, training_iter):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print('Training ExactGP model on {} iterations using {} kernel'.format(training_iter,name))
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_data)
        loss = -mll(output, train_labels)
        loss.backward()
        optimizer.step()
        print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
    print('')
    print(f'Training of MultiOutputExactGP model with {name} kernel complete!\n')

    # Save the model
    torch.save(model.state_dict(),f'models/MultiOutputExactGP/model-{name}.pth')
    torch.save(likelihood.state_dict(),f'models/MultiOutputExactGP/likelihood-{name}.pth')


def test_ExactMultiOutputExactGP__(test_data, test_labels, model, likelihood, name, T, save=False, show=True):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        observed_pred = model(test_data)
        lower, upper = observed_pred.confidence_region()
    
    # x-axis plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    fig.suptitle(f'MultiOutputExactGP Wind Estimation (x-axis) with {name} kernel')
    ax.plot(T,test_labels[0],color='orange',label='True Data')
    ax.plot(T,observed_pred.mean[:,0],'b-',label="Estimated Data")
    ax.fill_between(T, lower[:,0].numpy(), upper[:,0].numpy(), alpha=0.5, color='cyan',label='Confidence')
    ax.legend()
    if save:
        plt.savefig(f'imgs/gp_plots/MultiOutputExactGP/{name}-x',dpi=300)

    # y-axis plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    fig.suptitle(f'MultiOutputExactGP Wind Estimation (y-axis) with {name} kernel')
    ax.plot(T,test_labels[1],color='orange',label='True Data')
    ax.plot(T,observed_pred.mean[:,1],'b-',label="Estimated Data")
    ax.fill_between(T, lower[:,1].numpy(), upper[:,1].numpy(), alpha=0.5, color='cyan',label='Confidence')
    ax.legend()
    if save:
        plt.savefig(f'imgs/gp_plots/MultiOutputExactGP/{name}-y',dpi=300)

    if show:
        plt.show()
    plt.close()

