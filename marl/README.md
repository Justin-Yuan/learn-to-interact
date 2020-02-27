# MARL 

training commands 

```bash 
python main.py 
```


submodules 
reference: https://www.vogella.com/tutorials/GitSubmodules/article.html

add submodule to repo 
```bash
git submodule add -b master URL
git submodule init 
```

- to clone with submodules 
```bash 
git clone --recursive URL 
```

- to pull updates from `mpe_hierarchy` submodule, do 
```bash
git submodule update --remote
```

## Notes 

- runner scheme: make_env -> wrap in vec_env -> build runner 



## Pip install 
```bash
cat requirements.txt | xargs -n 1 pip install
```
