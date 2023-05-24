#include "message.h"

char* strCopy(const char* str)
{
	char *newStr = (char*)malloc(strlen(str) + 1);
	if (newStr) {
		strncpy(newStr, str, strlen(str) + 1);
	}
	else newStr = NULL;
	return newStr;
}

Message ParseMessage(const char* str)
{
	Message m;
	cJSON* node = cJSON_Parse(str);
	cJSON *role = cJSON_GetObjectItem(node, "r");
	cJSON* type = cJSON_GetObjectItem(node, "t");
	cJSON* message = cJSON_GetObjectItem(node, "m");
	cJSON* body = cJSON_GetObjectItem(node, "b");

	m.role = strCopy(role->valuestring);
	m.type = strCopy(type->valuestring);
	m.message = strCopy(message->valuestring);
	m.body = strCopy(body->valuestring);

	cJSON_Delete(node);
	return m;
}

cJSON* cJsonFromMessage(Message* m, int parse_body)
{
	cJSON* node = cJSON_CreateObject();
	cJSON_AddStringToObject(node, "r", strCopy(m->role));
	cJSON_AddStringToObject(node, "t", strCopy(m->type));
	cJSON_AddStringToObject(node, "m", strCopy(m->message));
	cJSON_AddStringToObject(node, "b", strCopy(m->body));
	return node;
}

cJSON* cJsonMessage(const char *role,const char *type,const char *message)
{
	cJSON* node = cJSON_CreateObject();
	cJSON_AddStringToObject(node, "r", role);
	cJSON_AddStringToObject(node, "t", type);
	cJSON_AddStringToObject(node, "m", message);
	cJSON_AddObjectToObject(node, "b");
	return node;
}

void FreeMessage(Message* m) 
{
	free(m->role);
	free(m->type);
	free(m->message);
	free(m->body);
}