all:
	echo "Don't run make without arguments"

format:
	isort .
	black .

push: format
	git add .
	git commit
	git push origin master
