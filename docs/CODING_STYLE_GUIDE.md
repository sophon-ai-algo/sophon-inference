## C++ Style Guide

### Style Guide
##### [Reference to Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
##### Modifications
* When declaring a pointer/reference variable or argument, you should place the asterisk adjacent to the type:
```shell
int x, *y;  // Disallowed - no & or * in multiple declaration
char * c;  // Bad - spaces on both sides of *
const string & str;  // Bad - spaces on both sides of &
char *c;  // Bad - spaces on left side of *
const string &str;  // Bad - spaces on left side of &

// Right exampels as following
char* c;
const string& str;
```

### [Style check tool for C++](https://github.com/cpplint/cpplint)
```shell
cpplint --filter='-build/namespaces,-legal/copyright,-runtime/references,-runtime/int' file_name.cpp
```

## Python Style Guide

### Style Guide
##### [Reference to Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
##### Modifications
* Indent your code blocks with 2 spaces.

### [Style check tool for Python](https://github.com/PyCQA/pylint)
```shell
# '.pylintrc' is in the project root path
pylint --rcfile='.pylintrc' file_name.py
```

## Shell Style Guide

### Style Guide
##### [Reference to Google Shell Style Guide](https://google.github.io/styleguide/shell.xml)

### [Style check tool for Shell Scripts](https://github.com/koalaman/shellcheck)
```shell
shellcheck file_name.sh
```
