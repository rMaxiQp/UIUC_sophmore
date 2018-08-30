/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

// Author: Aneesh Durg
// Key_codes:g
// F_name:my_ext
// Description:"Interactively inserts text"
//---
#include <string.h>
void my_ext(editor *editor, display *display, char **buffer, char k) {
    (void)buffer;
    (void)k;
    char *input = NULL;
    display_interact(display, "Text to insert:", &input);
    if (!input) {
        return;
    }
    if (input[strlen(input) - 1] == '\n')
        input[strlen(input) - 1] = 0;
    location loc = display_get_loc(display);
    handle_insert_command(editor, loc, input);
    free(input);
}
