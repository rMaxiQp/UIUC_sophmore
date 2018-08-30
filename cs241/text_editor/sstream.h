/**
* Text Editor Lab
* CS 241 - Spring 2018
*/

#pragma once
/**
 * In computer science, a stream is a sequence of data elements made available
 * over time. A stream can be thought of as items on a conveyer belt being
 * processed one at a time rather than in large batches.
 *
 * https://en.wikipedia.org/wiki/Stream_(computing)
 *
 * The string stream class (shortened to sstream) is a container that
 * implements a stream of characters, as well as some other features not
 * normally associated with a generic stream.
 *
 * The sstream stores an underlying buffer of characters and maintains a
 * "position" within this buffer. As characters are read from the stream,
 * the position of the stream advances forward by the number of characters
 * successfully read. Thus, multiple reads from the stream will generally
 * return different outputs. This position shall be exposed to and may be
 * modified by the user.
 *
 * The user may push additional sequences of characters to the sstream. By
 * default, these new characters shall be pushed to the end of the underlying
 * buffer without modifying the sstream's position.
 *
 * Note that, whenever a C-string is added to the string stream, the null byte
 * in that C-string is NOT considered to be added to the stream.
 */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Denotes maximum and minimum valid read counts */
#define SSTREAM_MAX LONG_MAX
#define SSTREAM_MIN LONG_MIN

/* Forward declare the structure */
typedef struct sstream sstream;

/* A bytestring is simply a character array accompanied by a size variable.
 * The size variable not only allows the size of the array to be checked in
 * O(1) time, but also allows storage of arbitrary characters, including
 * several '\0' (null) bytes. C-strings cannot contain multiple null bytes by
 * definition, since the first null byte delimits the end of a C-string.
 *
 * However, to maintain compatibility with C standard library functions such
 * as printf(), every bytestring produced by sstream methods shall contain an
 * extra null byte '\0' at the very end of its buffer, which does NOT
 * contribute to its size. bytestrings passed into sstream methods cannot
 * be expected to abide by this rule however, unless the `size` member is
 * negative.
 *
 * By convention, if the `size` of a bytestring passed into sstream methods is
 * negative, it indicates that `str` is a C-string whose length can be
 * calculated by finding the position of its null byte.
 */
typedef struct bytestring {
    char *str;
    ssize_t size;
} bytestring;

/**
 * Returns a new sstream allocated on the heap. The initial contents of the
 * sstream are determined by the following rules:
 *
 * -If `bytes.str` is NULL, creates an empty sstream
 * -If `bytes.str` is not NULL and `bytes.size` is negative, interpret
 *  `bytes.str` as a C-string and initialize the sstream to that C-string
 * -If `bytes.str` is not NULL and `bytes.size` is positive or zero, interpret
 *  `bytes.str` as a character array of `bytes.size` bytes and initialize the
 *  sstream thereto. `bytes.str` is guaranteed to have `bytes.size` bytes
 * available for you to read.
 *
 * Any implementation for an empty sstream `this` should result in
 * sstream_eos(this) evalutating to true and sstream_size(this) evaluating to 0
 *
 * In any case, the sstream shall not share any memory with the `bytes.str`
 * buffer. The initial position of the sstream shall always be 0.
 */
sstream *sstream_create(bytestring bytes);

/**
 * Destroys this sstream object, freeing all of its dynamic memory. `this`
 * shall be a valid sstream allocated on the heap.
 */
void sstream_destroy(sstream *this);

/**
 * Sets the internal buffer to the contents of `bytes`. `bytes.str` shall be
 * interpreted in a similar manner as prescribed by the constructor, i.e. as
 * either a C-string or an array of `bytes.size` bytes.
 *
 * The input sstream may be any valid sstream and all possible cases must be
 * handled.
 *
 * The positon of the sstream shall be reset to 0.
 */
void sstream_str(sstream *this, bytestring bytes);

/**
 * Returns whether the sstream is at its end (end of sstream). This indicates
 * that no more data can be read from this sstream until the sstream position is
 * modified, or until new data is assigned to the sstream.
 *
 * Note that, when the sstream is at its end, its position shall not reference
 * any valid character ever added to the sstream.
 */
bool sstream_eos(sstream *this);

/**
 * Returns the character located at the given offset from the current position
 * within the sstream. In other words, `sstream_peek(this, 0)` gives the
 * character at the current sstream position.
 *
 * This method does not advance the postion of the sstream.
 *
 * This method results in undefined behavior if a character at or beyond
 * the end of the sstream, or before the beginning, is peeked.
 */
char sstream_peek(sstream *this, ssize_t offset);

/**
 * Returns the character located at the current position within the sstream,
 * advancing the position by one.
 *
 * This method results in undefined behavior if the sstream is at its end.
 */
char sstream_getch(sstream *this);

/**
 * Returns the size of the sstream object's character sstream. The size
 * represents the number of characters actually inserted into the sstream
 * (minus all characters that were flushed out through erase), not the number
 * of characters remaining from the sstream position to the end. In other
 * words, repeated reads from the sstream alone shall not affect this measure.
 */
size_t sstream_size(sstream *this);

/**
 * Returns the sstream position.
 */
size_t sstream_tell(sstream *this);

/**
 * Updates the sstream position by `offset` bytes. The nature of this offset
 * is determined in the same manner as lseek(2) through the `whence` argument,
 * which may be SEEK_SET, SEEK_CUR, or SEEK_END.
 *
 * If the new position were to go beyond the bounds of the internal buffer,
 * i.e. if `new_pos > sstream_size(this) || new_pos < 0`, then returns -1 and
 * does not alter the sstream position.
 *
 * Returns 0 on success.
 */
int sstream_seek(sstream *this, ssize_t offset, int whence);

/**
 * Returns the number of characters in the sstream from the current position
 * to the end, including the character at the current position. When the
 * sstream is at its end, this method returns 0.
 */
size_t sstream_remain(sstream *this);

/**
 * Let `count_t` denote `min(abs(count), sstream_remain(this))` if
 * `count >= 0`, and `min(abs(count), sstream_tell(this))` if `count < 0`.
 * In other words, `count_t` is the actual number of bytes to be read.
 *
 * This method reads exactly `count_t` bytes from `this` and stores the result
 * in `out`, subject to some conditions:
 *   -If `out->str == NULL`, allocates a buffer of `count_t+1` bytes on the
 *   heap, sets `out->str` to this new buffer, and sets `out->size` to
 *   `count_t`
 *   -If `out->str != NULL`, then assumes `out->str` is a heap allocated buffer
 *   of at least `out->size + 1` bytes; iff `out->size < count_t`, resizes
 *   the buffer to `count_t + 1` bytes using realloc(3) and sets `out->size`
 *   to `count_t`.
 *   -In any case, set `out->str[count_t] = '\0'` for compatibility reasons
 *
 * If `count < 0`, then reads `count_t` bytes before the current sstream
 * position, NOT including the byte at the current position, and does NOT
 * advance the sstream position. Otherwise, reads `count_t` bytes after
 * the current position, including the byte at the current position, and
 * advances the position `count_t` bytes. Any reads made, regardless of
 * the sign of count, should be placed into `out->str` in left to right order.
 *
 * In any case, `out->str` shall hold the result of the read, and the number
 * of bytes read (i.e. `count_t`) shall be returned.
 */
size_t sstream_read(sstream *this, bytestring *out, ssize_t count);

/**
 * Append the contents of `bytes` to the end of this sstream. `bytes` will
 * either be interpreted as a C-string or a character array of length
 * `bytes.size`, subject to the same conditions as in `sstream_create`.
 *
 * This has no effect on the position, but will increase the size of the
 * sstream by the number of bytes read.
 */
void sstream_append(sstream *this, bytestring bytes);

/**
 * Finds the next occurrence of the target bytestring that occurs after the
 * sstream position and returns the offset from the current position at which
 * it is found. A return value of 0, for example, indicates that the target
 * bytestring occurs at the current sstream position.
 *
 * The rule for determining whether `bytes` represents a C-string or a raw
 * array of characters is the same as in the constructor.
 *
 * Returns -1 if the target bytestring cannot be found.
 */
ssize_t sstream_subseq(sstream *this, bytestring bytes);

/**
 * Removes at most the next `abs(number)` bytes from the sstream buffer.
 *
 * If `number` is positive, erases at most the next `number` bytes, including
 * the byte at the current position, without changing the absolute position
 * number. i.e. `sstream_tell(this)` shall not be changed.
 *
 * If `number` is negative, erases at most the previous `abs(number)` bytes,
 * excluding the byte at the current position, and decreases the absolute
 * position number by `min(abs(number), sstream_tell(this))`.
 * i.e. `sstream_peek(this)` shall not be changed.
 *
 * The size of sstream must be decreased by the number of bytes successfully
 * deleted.
 *
 * Returns the number of bytes successfully deleted.
 */
size_t sstream_erase(sstream *this, ssize_t number);

/**
 * Writes the `bytes` bytestring at the current sstream position,
 * overwriting existing characters in the sstream buffer up to the length of
 * the bytestring.
 *
 * Whether `bytes` is interpreted as a C-string or a character array
 * follows the same behavior as the `sstream_create` method.
 *
 * If need be, the sstream buffer (and size) shall be expanded to contain the
 * input bytestring. The sstream position shall not change.
 */
void sstream_write(sstream *this, bytestring bytes);

/**
 * Operates like `sstream_write`, except `bytes` is inserted into the buffer,
 * shifting everything at the sstream position and beyond back by the
 * effective length of `bytes`. This does not change the sstream position,
 * but will always increase the size by the number of bytes added.
 */
void sstream_insert(sstream *this, bytestring bytes);

/**
 * This function parses a long and stores it in `out`. This shall only
 * handle decimal numbers. A valid long consists of an optional '-'
 * followed by any nonzero number of decimal digits such that overflow would
 * not occur. If overflow would occur, it shall read as many digits as possible
 * before this would happen.
 *
 * Note that the long must be found at the current position, with no leading
 * whitespace. If no valid long can be parsed at the current position, then
 * parsing shall fail.
 *
 * If parsing was successful, the position advances by the number of characters
 * read as part of the long and returns this number. Otherwise, returns -1.
 */
int sstream_parse_long(sstream *this, long *out);
