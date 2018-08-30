/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

#pragma once
#include <stdio.h>

/**
 * This library will handle all the formating
 * for your document and text editor.
 *
 * Please use this to ensure that your formatting
 * matches what the autograder expects.
 */

/**
 * Call this function when your user has incorrectly
 * executed your text editor.
 */
void print_usage_error();

void invalid_command(char *command);

/**
 * Prints out one line from 'document' starting from 'start_col_index'. Only
 * 'max_cols' characters will be printed. If 'max_cols' == -1, all characters
 * starting from the 'start_col_index' will be printed.
 */
void print_line(document *document, size_t index, size_t start_col_index,
                ssize_t max_cols);

/**
 * Error message that should be displayed if the document is empty and the user
 * attempts to display somethin.
 */
void print_document_empty_error();
void found_search(char *search_str, size_t line_no, size_t idx);
void not_found(char *search_str);
