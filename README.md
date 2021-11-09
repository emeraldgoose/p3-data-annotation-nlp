# data-annotation-nlp-level3-nlp-10

## 1. DVC 설치

1. [dvc github](https://github.com/iterative/dvc/releases) 에서 `dvc_2.8.2_amd64.deb` 다운로드
2. `apt install ./dvc_2.8.2_amd64.deb`을 통해 package 설치
3. `git remote add {github_path}`로 git을 연결해준다.
4. `git pull`로 git을 가져온다.
5. `git checkout {dataset 브랜치}`로 checkout 한다
6. `dvc remote add --default gdrive://{google_drive}`로 gdrive와 연결해준다.
7. `dvc pull`을 한 후에, gdrive 구글 계정 인증을 해준다.
8. 데이터 생성!
