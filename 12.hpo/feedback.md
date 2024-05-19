### Helpful

In `_evaluate_config` function in `meta_hyper/api.py` file, you catch an exception logging 
```
"An error occured during evaluation of config {config_id}: " f"{config}."
```
More informative error strings would make user's life easier. For example. You may catch the exception as a variable (i.e. `catch Exception as e:`) then do some manipulations on it to give some information (e.g. type) of the exception. In asynchronous programming languages like JavaScript, people tend to use objects (dicts in Python) to store a variety of information. Instead of assigning a string like `result = "error` you may assign a dictionary object.

Thank you for the project AutoML, which saves substantial amount of time when doing HPO. Before that I was writing boilerplate code all the time. Even copying a boilerplate code becomes tedious after some time :D. 