import matplotlib.pyplot as plt
from ShockSR import ShockSR

rho1,rho5 = 1, 0.1
p1,p5 = 1e4, 10
t = 1

xi,rho,p,v = ShockSR(rho1,p1,rho5,p5)
x = xi*t

plt.figure(figsize=(6.4,8))

plt.subplot(3,1,1)
plt.plot(x,rho)
plt.ylabel(r'$\rho =$ density / $\rho_1$', fontsize=14)

plt.subplot(3,1,2)
plt.plot(x,p)
plt.ticklabel_format(axis='y', style='sci',
                     scilimits=(0,0), useMathText=True)
plt.ylabel(r'$p =$ pressure / $\rho_1c^2$', fontsize=14)

plt.subplot(3,1,3)
plt.plot(x,v)
plt.ylabel(r'$v =$ velocity / c', fontsize=14)
plt.xlabel(r'$\xi = x/t$ ', fontsize=14)

plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()

