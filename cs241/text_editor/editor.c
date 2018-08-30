/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

#include "document.h"
#include "editor.h"
#include "format.h"
#include "sstream.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CSTRING(x) ((bytestring){(char *)x, -1})
#define DECLARE_BUFFER(name) bytestring name = {NULL, 0};
#define EMPTY_BYTES ((bytestring){NULL, 0})
#define min(a,b) (a <= b ? a : b)

char *get_filename(int argc, char *argv[]) {
    // TODO implement get_filename
    // take a look at editor_main.c to see what this is used for
    return argv[argc - 1];
}

sstream *handle_create_stream() {
    // TODO create empty strea
    return sstream_create(EMPTY_BYTES);
}
document *handle_create_document(const char *path_to_file) {
    // TODO create the document
    return document_create_from_file(path_to_file);
}

void handle_cleanup(editor *editor) {
    // TODO destroy the document
    document_destroy(editor->document);
}

void handle_display_command(editor *editor, size_t start_line,
                            ssize_t max_lines, size_t start_col_index,
                            ssize_t max_cols) { //check
    // TODO implement handle_display_command
    //assert(editor);
    //assert(start_line);

    if(!editor->document || !document_size(editor->document)) {
       print_document_empty_error();
       return;
    }
    size_t size = document_size(editor->document);

    if((max_lines != -1) && (size > max_lines + start_line - 1))
       size = max_lines + start_line - 1;
    for(size_t t = start_line; t <= size; t++)
       print_line(editor->document, t, start_col_index, max_cols);
}

void handle_insert_command(editor *editor, location loc, const char *line) {
    // TODO implement handle_insert_command
    assert(editor);

    while(loc.line_no > document_size(editor->document)){
       document_insert_line(editor->document, loc.line_no, "\0");
    }

    const char* line_doc = document_get_line(editor->document, loc.line_no);
    char* buffer = calloc(1, (strlen(line) + strlen(line_doc) + 1) * sizeof(char));
    strncpy(buffer, line_doc, loc.idx);
    strcat(buffer + loc.idx, line);
    strcat(buffer + loc.idx + strlen(line), line_doc + loc.idx);
    buffer[strlen(buffer)] =0;
    document_set_line(editor->document, loc.line_no, buffer);
    free(buffer);
}


void handle_append_command(editor *editor, size_t line_no, const char *line) { //check
    // TODO implement handle_append_command
    assert(editor);
    while(line_no > document_size(editor->document)) {
       document_insert_line(editor->document, line_no, "\0");
    }
    sstream_str(editor->stream, CSTRING(document_get_line(editor->document, line_no)));
    char* line_copy = strdup(line);
    char* ptr_to_char = strchr(line_copy, '\\');
    if(!ptr_to_char) {
       sstream_append(editor->stream, CSTRING(line_copy));
       free(line_copy);
       DECLARE_BUFFER(copy);
       sstream_seek(editor->stream, 0, SEEK_SET);
       sstream_read(editor->stream, &copy,sstream_size(editor->stream));
       document_set_line(editor->document, line_no,  copy.str);
       free(copy.str);
       return;
    }

    ptr_to_char = line_copy;
    int first = 1;
    char* prev = ptr_to_char;

    while(*(ptr_to_char)) {
       if(*(ptr_to_char) == '\\' && *(ptr_to_char + 1) == 'n') {
          if(first) {
             first = 0;
             char* str = strndup(line_copy, ptr_to_char - line_copy);
             sstream_append(editor->stream, CSTRING(str));
             //printf("str:: %s\n", str);
             free(str);
             DECLARE_BUFFER(copy);
             sstream_read(editor->stream, &copy, sstream_size(editor->stream));
             document_set_line(editor->document, line_no, copy.str);
             free(copy.str);
             prev = ptr_to_char  + 2;
          }
          else if(ptr_to_char == prev) {
             document_insert_line(editor->document, line_no, "\0");
             line_no++;
             prev = ptr_to_char + 2;
          }
          else {
             char* str = strndup(line_copy + (prev - line_copy), ptr_to_char - prev);
             document_insert_line(editor->document, line_no, str);
             free(str);
             line_no++;
             prev = ptr_to_char + 2;
          }
       }
       ptr_to_char++;
    }
    if(ptr_to_char > prev) {
       char* str = strndup(line_copy + (prev - line_copy), ptr_to_char - prev);
       if(first)
          document_set_line(editor->document, line_no, str);
       else
          document_insert_line(editor->document, line_no, str);
       free(str);
    }
    //else if(*(ptr_to_char -1) == 'n' && *(ptr_to_char) == '\\')
     //  document_insert_line(editor->document, line_no,"");
    free(line_copy);
}

void handle_write_command(editor *editor, size_t line_no, const char *line) {
    // TODO implement handle_write_command
    assert(editor);

    while(line_no > document_size(editor->document)) {
       document_insert_line(editor->document, line_no, "");
    }

    sstream_str(editor->stream, CSTRING(document_get_line(editor->document, line_no)));
    char* line_copy = strdup(line);
    char* ptr_to_char = strchr(line_copy, '\\');
    if(!ptr_to_char) {
       document_set_line(editor->document, line_no, line_copy);
       return;
    }

    ptr_to_char = line_copy;
    int first = 1;
    char* prev = ptr_to_char;

    while(*(ptr_to_char)) {
       if(*(ptr_to_char) == '\\' && *(ptr_to_char + 1) == 'n') {
          if(first) {
             first = 0;
             char* str = strndup(line_copy, ptr_to_char - line_copy);
             document_set_line(editor->document, line_no, str);
             free(str);
             prev = ptr_to_char  + 2;
          }
          else if(prev == ptr_to_char) {
             document_insert_line(editor->document, line_no, "\0");
             line_no++;
             prev = ptr_to_char + 2;
          }
          else {
             char* str = strndup(line_copy + (prev - line_copy), ptr_to_char - prev);
             document_insert_line(editor->document, line_no, str);
             free(str);
             prev = ptr_to_char + 2;
             line_no++;
          }
       }
       ptr_to_char++;
    }

    if(ptr_to_char > prev) {
       char* str = strndup(line_copy + (prev - line_copy), ptr_to_char - prev);
       if(first)
          document_set_line(editor->document, line_no, str);
       else
          document_insert_line(editor->document, line_no, str);
       free(str);
    }
    //else if(*(ptr_to_char -1) == 'n' && *(ptr_to_char) == '\\')
      // document_insert_line(editor->document, line_no,"");

    free(line_copy);
}


void handle_delete_command(editor *editor, location loc, size_t num_chars) {
    // TODO implement handle_delete_command
    assert(editor);

    if(loc.line_no > document_size(editor->document)) {//clean
       for(size_t t = 0; t < document_size(editor->document); t++) {
          document_delete_line(editor->document, 0);
       }
    }
    sstream_str(editor->stream, CSTRING(document_get_line(editor->document, loc.line_no)));
    if(num_chars + loc.idx > sstream_size(editor->stream))
       num_chars = sstream_size(editor->stream) - loc.idx;

    sstream_seek(editor->stream, loc.idx, SEEK_SET);
    sstream_erase(editor->stream, num_chars);
    DECLARE_BUFFER(buffer);
    sstream_seek(editor->stream, 0, SEEK_SET);
    sstream_read(editor->stream, &buffer, sstream_size(editor->stream));
    document_set_line(editor->document, loc.line_no, buffer.str);
    free(buffer.str);
}

void handle_delete_line(editor *editor, size_t line_no) {
    // TODO implement handle_delete_line
    if(line_no > document_size(editor->document)) {
       document_delete_line(editor->document, document_size(editor->document));
    }
    else {
       document_delete_line(editor->document, line_no);
    }
}

location handle_search_command(editor *editor, location loc,
                               const char *search_str) {
    // TODO implement handle_search_command
    if(search_str == 0)
       return (location){0,0};
    size_t cur_line_no = loc.line_no - 1;
    size_t cur_idx = loc.idx;
    size_t size = document_size(editor->document);

    for(size_t t = 0; t <= size; t++) {
       cur_line_no = (cur_line_no + 1) % size;
       if(cur_line_no == 0) cur_line_no = size;

       const char* line = document_get_line(editor->document, cur_line_no);
       //printf("get: %s with %zu\n", line, t);
       //print_line(editor->document, cur_line_no, 0, -1);
       char* result = strstr(line + cur_idx, search_str);
       if(result)
          return (location){cur_line_no, result-line};
       cur_idx = 0;
    }
    return (location){0,0};
}

void handle_merge_line(editor *editor, size_t line_no) {
    // TODO implement handle_merge_line
    if(line_no >= document_size(editor->document)) return;
    sstream_str(editor->stream, CSTRING(document_get_line(editor->document, line_no)));
    sstream_append(editor->stream, CSTRING(document_get_line(editor->document, line_no + 1)));
    bytestring* buffer = malloc(sizeof(bytestring));
    buffer->str = NULL;
    buffer->size = -1;
    sstream_read(editor->stream, buffer, sstream_size(editor->stream));
    document_set_line(editor->document, line_no, buffer->str);
    document_delete_line(editor->document, line_no+1);
    free(buffer->str);
    free(buffer);
}

void handle_split_line(editor *editor, location loc) {
    // TODO implement handle_split_line
    const char* string = document_get_line(editor->document, loc.line_no);
    size_t size = strlen(string) - loc.idx + 1;
    char* split_line = malloc(size);
    memcpy(split_line, string + loc.idx, size);
    document_insert_line(editor->document, loc.line_no, split_line);
    handle_delete_command(editor, loc, strlen(split_line));
    free(split_line);
}

void handle_save_command(editor *editor) {
    // TODO implement handle_save_command
    document_write_to_file(editor->document, editor->filename);
}
