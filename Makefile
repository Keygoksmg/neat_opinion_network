lab:
	jupyter lab --no-browser --port=9999
activate_venv:
	source neat/bin/activate
a:
	echo $(ls)
screen_kill:
	screen -X -S [session # you want to kill] quit