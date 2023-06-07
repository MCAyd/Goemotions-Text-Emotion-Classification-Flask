# locust -f locustfile.py --host=http://localhost:5000
# http://0.0.0.0:8089/

import time
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def perform_request(self):
        self.client.post("/", data={"content": "Glad you enjoyed man, yeah it was pretty spontaneous too, I was praying I wasnt about to pull some bartard move and get jammed lol."})

if __name__ == "__main__":
    MyUser().run()