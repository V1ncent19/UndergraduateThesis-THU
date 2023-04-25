import torch
from scipy import integrate
import scipy.stats as st
import math

def MLEloss(y, GMMparam, epsilon = 1e-10): 
    '''
    use in training
    '''
    out_pi, out_sigma, out_mu = GMMparam
    expand_y = y.reshape((-1,1)).tile((1,out_pi.size()[1]))
    result = torch.sum(out_pi/out_sigma*torch.exp(-((expand_y-out_mu)/out_sigma)**2/2), dim = 1, keepdim=True) + epsilon
    return torch.mean(-torch.log(result))



# def normalKLloss(y_true, GMMparam, a = None, b = None):
#     '''
#     used in testing, where we know the ground truth distribution of error as standard normal
#     y_obs \sim Gauss(y; y_true, 1)
#     '''
#     out_pi, out_sigma, out_mu = GMMparam
#     y_true =  y_true.tolist()[0]
#     if a == None or b == None:
#         a = min(out_mu.tolist()+[y_true])-5*max(out_sigma.tolist())
#         b = max(out_mu.tolist()+[y_true])+5*max(out_sigma.tolist())
    
#     def q(y):
#         qy = 0
#         for pi, sigma, mu in zip(out_pi.tolist(), out_sigma.tolist(), out_mu.tolist()):
#             qy += pi*st.norm.pdf(y, loc = mu, scale = sigma)
#         return qy
#     def plogpq(y):
#         return st.norm.pdf(y, loc = y_true, scale = 1)*math.log( st.norm.pdf(y, loc = y_true, scale = 1)/q(y) )
#     return integrate.quad(plogpq,a,b)[0]

# def KLloss(y_true, GMMparam, err_pdf_fun, a = None, b = None, epsilon = 1e-8):
#     '''
#     used in testing, where we know the ground truth distribution of error 
#     y_obs \sim y_true + err_pdf_fun()
#     y: (bs)
#     GMMparam: [(bs,kmix), (bs,kmix), (bs,kmix)]
#     '''
#     total_loss = []
#     out_pi, out_sigma, out_mu = GMMparam
#     for yt, p, s, m in zip(y_true.tolist(), out_pi.tolist(), out_sigma.tolist(), out_mu.tolist()):
#         try:
#             a = min(m+yt)-5*max(s)
#             b = max(m+yt)+5*max(s)
            
#             def q(y):
#                 qy = 0
#                 for pi, sigma, mu in zip(p, s, m):
#                     qy += pi*st.norm.pdf(y, loc = mu, scale = sigma)
#                 return qy
#             def plogpq(y):
#                 return err_pdf_fun(y, yt)*math.log( err_pdf_fun(y, yt)/(q(y)+epsilon) )
#             l = integrate.quad(plogpq,a,b) # l[0] = integration, l[1] = calculation error
#             if l[1] < 1e-3:
#                 total_loss.append(l[0])
#             else:
#                 print(f'outlier dropped {l}')
#         except:
#             print('Integration raised math error; skipped.')
#             continue
    
#     if len(total_loss) == 0:
#         return None
#     else: 
#         return sum(total_loss)/len(total_loss)
       


if __name__ == '__main__':
    print(f'testing MLEloss():')
    # GMMparam = ( torch.tensor([[0.1192, 0.8808],
    #     [0.2689, 0.7311],
    #     [0.0474, 0.9526]]), 
    #     torch.tensor([[2980.9580,   20.0855],
    #     [1096.6332, 7.3891],
    #     [   7.3891, 1096.6332]]), 
    #     torch.tensor([[5., 2.],
    #     [6., 4.],
    #     [5., 3.]]) )
    # test_y = torch.tensor([3,7,10])
    # print(f'GMMparam: {GMMparam}, test_y: {test_y}')
    # print(f'MLEloss: {MLEloss(test_y,GMMparam )}')
    # print(f"{MLEloss(torch.tensor([[1],[2]]), [ torch.tensor([[1],[1]]), torch.tensor([[1],[1]]), torch.tensor([[1],[2]]) ])}")
    # print(f'{KLloss(torch.tensor([[1],[2]]), [ torch.tensor([[1],[1]]), torch.tensor([[1],[1]]), torch.tensor([[1],[2]]) ], st.norm.pdf)}')
    print(f'{KLloss(torch.tensor([[0]]), [ torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[1]]) ], st.norm.pdf)}')


