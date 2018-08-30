#/usr/bin/env sh
wget -nc -r -l1 http://illinois.edu/
perl -p -i -e 's/(Students|Illinois|Illini|International|Research|College|Undergraduate|Graduate|Campus|School|Faculty|Alumni)/CS241/g' illinois.edu/index.html illinois.edu/*/*.html
echo 
echo "Add the following to your /etc/hosts file:"
echo "www.bbc.com 127.0.0.1"
echo "Then start your favorite webserver e.g."
echo "cd illinois.edu  and,  python -m SimpleHTTPServer  or to run port 80, sudo python -m SimpleHTTPServer 80"

