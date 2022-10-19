import git
import os

#stores the git hash of the git submodules
def main():
    repo1 = git.Repo(f'{os.curdir}/src/yolov5')
    data = (
        f"/src/yolov5 {repo1.head.object.hexsha}{chr(10)}"
    )
    with open("src/submod_git", "w") as f:
        f.write(data)

if __name__ == '__main__':
    main()