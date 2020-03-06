#include <dirent.h>
#include <unistd.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <errno.h>

int main()
{
  char* argv[] = { "foobar", "arg1", NULL };
  char* envp[] = { NULL };
  DIR *fd;

  fd = opendir("/bin");
  if (fd == NULL) {
      perror("execveat");
      exit(EXIT_FAILURE);
  }
  /* int execveat(int dirfd, const char *pathname,
                  char *const argv[], char *const envp[],
                  int flags); */
  /* Check execveat called with the correct arguments works. */
  if (execveat(fd, "echo", argv, envp, 0) == -1) {
      perror("execveat");
      exit(EXIT_FAILURE);
  }

  /* Check valgrind will produce expected warnings for the
     various wrong arguments. */
  do {
      char *mem = malloc(16);
      void *t = (void *) &mem[0];
      void *z = (void *) -1;
      int flag = *((int *) &mem[8]);

      execveat(-1, "/bin/echo", argv, envp, 0)
      execveat(-1, "echo", argv, envp, 0);
      execveat(fd, "echo", argv, envp, flag);
      execveat(fd, "", argv, envp, 0);
      execveat(fd, NULL, argv, envp, 0);
      execveat(fd, "echo", t, envp, 0);
      execveat(fd, "echo", z, envp, 0);
  } while (0);

    closedir(fd);
    exit(EXIT_SUCCESS);
}

