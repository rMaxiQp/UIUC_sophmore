/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

#include "document.h"
#include "editor.h"
#include "format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * You can programatically test your text editor.
*/
int main() {
    // Setting up a docment based on the file named 'filename'.
    char *filename = "test.txt";
    editor *ed = malloc(sizeof(editor));
    handle_create_stream();
    document* doc = handle_create_document(filename);
    ed->document = doc;
    handle_display_command(ed, 0, 20, 0 ,-1);
    //ed->document = doc;
    document_insert_line(doc, 5, "");
    size_t t = document_size(doc);
    //handle_delete_line(ed, 2);
    printf("%zu\t", t);
    t = document_size(doc);
    printf("%zu\n", t);
    handle_display_command(ed, 0, -1, 0, -1);
    //printf("+++insert+++\n");
    //handle_insert_command(ed,(location){2,3}, "WHATEVER>>>SIGH");
    //printf("+++write+++\n");
    //handle_write_command(ed, 3, "OVERWTIING");
   // printf("+++delete+++\n");
   // handle_delete_command(ed, (location){4,2}, 6);
    //printf("+++delete longer+++\n");
    //handle_delete_line(ed, 5);
    printf("+++search+++\n");
    handle_search_command(ed, (location){10,7}, "include");
    handle_display_command(ed, 1, 80, 0, -1);
   // document_write_to_file(doc, filename);
    handle_cleanup(ed);
    sstream_destroy(ed->stream);
    //document_destroy(doc);
    free(ed);
}
