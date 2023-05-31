#ifndef MESSAGE_H
#define MESSAGE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "cJSON.h"

typedef struct Message
{
	const char *role;
	const char *type;
	const char *message;
	const char *body;
}Message;

typedef struct AgentInfo {
    const char *name;
	const char* mac;
	const char* ip;
    int port;
}AgentInfo;

Message ParseMessage(const char* str);
cJSON* cJsonFromMessage(Message *m, int parse_body);
void FreeMessage(Message *m);
char* strCopy(const char* str);
cJSON* cJsonMessage(const char* role, const char* type, const char* message);
cJSON* cJsonAInfo(const char* name, const char* mac, const char* ip, int port);
AgentInfo ParseAInfo(cJSON* elem);
void FreeAInfo(AgentInfo* ai);
#ifdef __cplusplus
}
#endif

#endif