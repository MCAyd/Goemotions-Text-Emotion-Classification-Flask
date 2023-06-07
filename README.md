# nlpcw

## Getting started

First make sure run DistilBERT_Final notebook to save saved_model torch model to local. Then, you'll be able to run the Flask application.

To make run the application, go to the main folder and;

```
python3 run.py
```

Once you run the Flask application, in order to use the terminal to get classification probabilities on terminal page;
You can change the "content = ....."

```
curl -X POST -d "content=Glad you enjoyed man, yeah it was pretty spontaneous too, I was praying I wasnt about to pull some bartard move and get jammed lol." http://localhost:5000/
```

To make some stretch testing (Locust), while the application running, use;
(It will use http://localhost:8009)

```
locust -f locustfile.py --host=http://localhost:5000
```
