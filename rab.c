#include "TestU01.h"
#include <sys/stat.h>

int main (int argc, char *argv[])
{
    struct stat st;
    if(stat(argv[1], &st) == 0) {
        swrite_Basic = FALSE;
        bbattery_RabbitFile (argv[1], st.st_size);
    } else {
        printf("Error: Couldn't determine the size of the file %s\n", argv[1]);
        return 1;
    }
    return 0;
}
