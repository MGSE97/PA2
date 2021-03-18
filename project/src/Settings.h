#pragma once

#define STREAMS 16

#define BLOCK_DIM 512
#define BLOCK_DIM_2D 32

//#define LINES 128
#define LINES 1024
//#define LINES 2048
//#define LINES 4096
//#define LINES 8196
#define ITEM_SIZE 4096
#define LINE_PARTS 7
#define LINE_SIZE LINE_PARTS * ITEM_SIZE

#define EMPTY_CHAR 205

#define TEXT_RESULT_VALUES 2
#define TEXT_RESULT_VALUES_SIZE TEXT_RESULT_VALUES*ITEM_SIZE
#define RESULT_VALUES 21
#define RESULT_TYPE unsigned int