import L1_LinearActuation as act
import time as t

while 1:
	u_in = int(input('Enter 1 for extension, anything else for retraction.'))
	initial_t = t.time()
	act.actuate(u_in)
	wait_for_user = input('Press enter when fully extended.')
	time_calc = t.time() - initial_t
	print(time_calc)
	u_in = input("Press enter to retract.")
	initial_t = t.time()
	act.actuate()
	u_in = input("press enter when the arm is fully retracted.")
	time_calc = t.time() - initial_t
	print(time_calc)
	print('Exiting.')
	break
