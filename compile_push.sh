# remove old version file
echo "Delete old version file"
rm -r dist

# compile
echo "Start compile"
python3 setup.py sdist bdist_wheel

# push
echo "Start push ..."
python3 -m twine upload dist/*
