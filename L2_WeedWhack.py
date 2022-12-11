import L1_LinearActuation as act
import L1_WeedeaterRelay as weed
import time as t

def engage():
    act.actuate(1)
    t.sleep(9)
    weed.setRelay(1)
    t.sleep(5)
    weed.setRelay(0)
    act.actuate(0)
    t.sleep(9)

def weedEater_test():
    weed.setRelay(1)
    t.sleep(1)
    weed.setRelay(0)



if __name__ == "__main__":
    try:
        u_in = input('press enter to run weed eating script')
        engage()
        print('Exiting.')
    except KeyboardInterrupt:
        weed.setRelay(0)
        act.actuate(0)
        print('Exiting on KBI.')
