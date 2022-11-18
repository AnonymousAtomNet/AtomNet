/**
 ******************************************************************************
 * @file    aiPbMgr.c
 * @author  MCD/AIS Team
 * @brief   Helper function for AI ProtoBuffer support
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2019,2021 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#include <aiPbMgr.h>
#include <aiPbIO.h>

#include <pb_encode.h>
#include <pb_decode.h>

#include <aiTestUtility.h> /* for HAL and specific device functions */


/*---------------------------------------------------------------------------*/

typedef enum _pbMmgState {
  PB_MGR_NOT_INITIALIZED = 0,
  PB_MGR_READY = 1,
  PB_MGR_ON_GOING = 2,
} pbMgrState;

static struct pbContextMgr {
  pb_istream_t input;
  pb_ostream_t output;
  const aiPbCmdFunc *funcs;
  uint32_t  n_func;
  reqMsg  req;
  respMsg resp;
  pbMgrState state;
} pbContextMgr;

void aiPbMgrInit(const aiPbCmdFunc *funcs)
{
  const aiPbCmdFunc *cfunc;

  memset(&pbContextMgr, 0, sizeof(struct pbContextMgr));

  pb_io_stream_init();

  pbContextMgr.input = pb_io_istream(0);
  pbContextMgr.output = pb_io_ostream(0);

  pbContextMgr.n_func = 0;
  pbContextMgr.funcs = NULL;

  if (funcs) {
    cfunc = funcs;
    while (cfunc->process) {
      pbContextMgr.n_func++;
      cfunc++;
    }
    pbContextMgr.funcs = funcs;
  }

  pbContextMgr.state = PB_MGR_READY;
}

int aiPbMgrWaitAndProcess(void)
{
  uint32_t idx;
  const aiPbCmdFunc *cfunc;

  pb_io_flush_istream();
  if (pb_decode_delimited(&pbContextMgr.input, reqMsg_fields, &(pbContextMgr.req))) {
    pb_io_flush_istream();
    pbContextMgr.state = PB_MGR_ON_GOING;
    for (idx = 0; idx < pbContextMgr.n_func; idx++) {
      cfunc = &pbContextMgr.funcs[idx];
      if (cfunc->cmd == pbContextMgr.req.cmd) {
        cfunc->process(&(pbContextMgr.req), &(pbContextMgr.resp), cfunc->param);
        break;
      }
    }
    if (idx == pbContextMgr.n_func) {
      aiPbMgrSendAck(&(pbContextMgr.req), &(pbContextMgr.resp), EnumState_S_ERROR,
          EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
    }
    pbContextMgr.state = PB_MGR_READY;
  }

  pb_io_flush_istream();

  return 0;
}


/*---------------------------------------------------------------------------*/

void aiPbMgrSendResp(const reqMsg *req, respMsg *resp,
    EnumState state)
{
  resp->reqid = req->reqid;
  resp->state = state;
  pb_encode(&pbContextMgr.output, respMsg_fields, resp);
  pb_io_flush_ostream();
}

void aiPbMgrSendAck(const reqMsg *req, respMsg *resp,
    EnumState state, uint32_t param, EnumError error)
{
  resp->which_payload = respMsg_ack_tag;
  resp->payload.ack.param = param;
  resp->payload.ack.error = error;
  aiPbMgrSendResp(req, resp, state);
}

bool aiPbMgrWaitAck(void)
{
  bool res;
  ackMsg ack = ackMsg_init_default;
  res = pb_decode_delimited(&pbContextMgr.input, ackMsg_fields, &ack);
  pb_io_flush_istream();
  return res;
}

bool aiPbMgrSendLog(const reqMsg *req, respMsg *resp,
    EnumState state, uint32_t lvl, const char *str)
{
  bool res;
  ackMsg ack = ackMsg_init_default;

  size_t len = strlen(str);

  resp->which_payload = respMsg_log_tag;
  resp->payload.log.level = lvl;
  if (len >= sizeof(resp->payload.log.str))
    len = sizeof(resp->payload.log.str) - 1;

  memcpy(&resp->payload.log.str[0], str, len+1);

  aiPbMgrSendResp(req, resp, state);

  res = pb_decode_delimited(&pbContextMgr.input, ackMsg_fields, &ack);
  pb_io_flush_istream();
  return res;
}

bool aiPbMgrSendLogV2(EnumState state, uint32_t lvl, const char *str)
{
  bool res;
  ackMsg ack = ackMsg_init_default;

  if (pbContextMgr.state != PB_MGR_ON_GOING)
    return false;

  size_t len = strlen(str);

  pbContextMgr.resp.which_payload = respMsg_log_tag;
  pbContextMgr.resp.payload.log.level = lvl;
  if (len >= sizeof(pbContextMgr.resp.payload.log.str))
    len = sizeof(pbContextMgr.resp.payload.log.str) - 1;

  memcpy(&pbContextMgr.resp.payload.log.str[0], str, len+1);

  aiPbMgrSendResp(&(pbContextMgr.req), &(pbContextMgr.resp), state);

  res = pb_decode_delimited(&pbContextMgr.input, ackMsg_fields, &ack);
  pb_io_flush_istream();
  return res;
}

struct aiPbMgrBuffer {
  ai_buffer *buffer;
  uint32_t n_max;
  uint32_t n_ops;
  uint32_t err;
  void *msg;
};

uint32_t aiPbAiBufferSize(const ai_buffer *buffer)
{
  if (!buffer)
    return 0;
  else
    return buffer->size;
}


static ai_buffer_format aiPbMsgFmtToAiFmt(const uint32_t msgFmt)
{
  return (ai_buffer_format)msgFmt;
}

static uint32_t aiPbAiFmtToMsgFmt(const ai_buffer_format aiFmt)
{
  return (uint32_t)aiFmt;
}

static void ai_init_buffer_shape_msg(const ai_buffer *ai_buffer, aiBufferShapeMsg *shape)
{
  shape->format = aiPbAiFmtToMsgFmt(ai_buffer->format);
  shape->n_batches = AI_BUFFER_SHAPE_ELEM(ai_buffer, AI_SHAPE_BATCH);
  shape->height = AI_BUFFER_SHAPE_ELEM(ai_buffer, AI_SHAPE_HEIGHT);
  shape->width = AI_BUFFER_SHAPE_ELEM(ai_buffer, AI_SHAPE_WIDTH);
  shape->channels = AI_BUFFER_SHAPE_ELEM(ai_buffer, AI_SHAPE_CHANNEL);
  if (AI_BUFFER_SHAPE_SIZE(ai_buffer) > 4)
    shape->depth = AI_BUFFER_SHAPE_ELEM(ai_buffer, AI_SHAPE_DEPTH);
  else
    shape->depth = 0;
  if (AI_BUFFER_SHAPE_SIZE(ai_buffer) > 5)
    shape->extension = AI_BUFFER_SHAPE_ELEM(ai_buffer, AI_SHAPE_EXTENSION);
  else
    shape->extension = 0;
  shape->addr = (uint32_t)ai_buffer->data;
  shape->scale = 0.0;
  shape->zeropoint = 0;
}

static bool aiPbBuffer_read_cb3(pb_istream_t *stream, const pb_field_t *field,
    void **arg)
{
  struct aiPbMgrBuffer *bm = (struct aiPbMgrBuffer *)*arg;
  aiBufferByteMsg *msg;
  ai_buffer_format format;
  size_t itsize;

  UNUSED(field);

  const ai_u32 b_ = AI_BUFFER_SHAPE_ELEM(bm->buffer, AI_SHAPE_BATCH);
  const ai_u32 h_ = AI_BUFFER_SHAPE_ELEM(bm->buffer, AI_SHAPE_HEIGHT);
  const ai_u32 w_ = AI_BUFFER_SHAPE_ELEM(bm->buffer, AI_SHAPE_WIDTH);
  const ai_u32 c_ = AI_BUFFER_SHAPE_ELEM(bm->buffer, AI_SHAPE_CHANNEL);
  const ai_u32 d_ = (AI_BUFFER_SHAPE_SIZE(bm->buffer) > 4)?AI_BUFFER_SHAPE_ELEM(bm->buffer, AI_SHAPE_DEPTH):0;
  const ai_u32 e_ = (AI_BUFFER_SHAPE_SIZE(bm->buffer) == 6)?AI_BUFFER_SHAPE_ELEM(bm->buffer, AI_SHAPE_EXTENSION):0;

  int maxr = bm->n_max;  /* number of item w/o padding */
  msg = (aiBufferByteMsg *)bm->msg;
  format = aiPbMsgFmtToAiFmt(msg->shape.format);

  /* Check shape/format */
  bm->err = EnumError_E_NONE;
  if ((format == AI_BUFFER_FORMAT_NONE) || (format != bm->buffer->format)) {
    maxr = 0;
    bm->err = EnumError_E_INVALID_FORMAT;
  } else if ((msg->shape.channels != c_) ||
      (msg->shape.height != h_) ||
      (msg->shape.width != w_) ||
      (msg->shape.n_batches != b_) ||
      (msg->shape.depth != d_) ||
      (msg->shape.extension != e_))
  {
    maxr = 0;
    bm->err = EnumError_E_INVALID_SHAPE;
  }

  itsize = 1; /* byte-by-byte read mode by default,
                 can be optimized, however data are exchanged with the host
                 by 32bytes, pb_read() read the data in a local buffer. */
  if (maxr) {
#if defined(TFLM_RUNTIME) && TFLM_RUNTIME == 1
    maxr = aiPbAiBufferSize(bm->buffer) * AI_BUFFER_FMT_GET_BITS(format) / 8;
#else
    maxr = AI_BUFFER_BYTE_SIZE(AI_BUFFER_SIZE(bm->buffer) * b_, format);
#endif
    bm->n_max = maxr;
  }

  /* Read data */
  uint8_t *pw = (uint8_t *)bm->buffer->data;
  while (stream->bytes_left) {
    uint64_t number;
    if (!pb_read(stream, (pb_byte_t *)&number, itsize))
      return false;
    if (maxr > 0) {
      if (pw) {
        // memcpy(pw, &number, itsize);
        *pw = *(uint8_t *)&number;
        pw += itsize;
      }
      maxr--;
    }
    bm->n_ops++;
  }

  /* Check nb_op */
  if ((bm->err == EnumError_E_NONE) && (bm->n_ops != bm->n_max))
    bm->err = EnumError_E_INVALID_SIZE;

  return true;
}

static bool aiPbBuffer_write_cb3(pb_ostream_t *stream, const pb_field_t *field,
    void * const *arg)
{
  struct aiPbMgrBuffer *bm = (struct aiPbMgrBuffer *)*arg;
  size_t n_bytes;

  int maxw = bm->n_max;
  ai_buffer_format format;

  if ((maxw == 0) || (!bm->buffer))
    return true;

  format = bm->buffer->format;
#if defined(TFLM_RUNTIME) && TFLM_RUNTIME == 1
  n_bytes = aiPbAiBufferSize(bm->buffer) * AI_BUFFER_FMT_GET_BITS(format) / 8;
#else
  const ai_u32 b_ = AI_BUFFER_SHAPE_ELEM(bm->buffer, AI_SHAPE_BATCH);
  n_bytes = AI_BUFFER_BYTE_SIZE(AI_BUFFER_SIZE(bm->buffer) * b_, format);
#endif

  /* Write data */
  pb_byte_t *pr = (pb_byte_t *)bm->buffer->data;

  if (!pb_encode_tag_for_field(stream, field))
    return false;

  if (!pb_encode_string(stream, pr, n_bytes))
    return false;

  bm->n_ops = maxw;

  return true;
}

bool aiPbMgrReceiveAiBuffer3(const reqMsg *req, respMsg *resp,
    EnumState state, ai_buffer *buffer)
{
  aiBufferByteMsg msg;
  struct aiPbMgrBuffer hdlb;
  bool res = true;

  hdlb.n_ops = 0;
  hdlb.buffer = buffer;
  hdlb.err = EnumError_E_NONE;
  hdlb.n_max = aiPbAiBufferSize(buffer);
  hdlb.msg = &msg;

  msg.datas.funcs.decode = &aiPbBuffer_read_cb3;
  msg.datas.arg = &hdlb;

  /* Waiting buffer message */
  pb_decode_delimited(&pbContextMgr.input, aiBufferByteMsg_fields, &msg);
  pb_io_flush_istream();

  /* Send ACK and wait ACK (or send ACK only if error) */
  if (hdlb.err) {
    aiPbMgrSendAck(req, resp, EnumState_S_ERROR, hdlb.err,
        (EnumError)hdlb.err);
    res = false;
  } else {
    aiPbMgrSendAck(req, resp, state, hdlb.n_ops, EnumError_E_NONE);
    if ((state == EnumState_S_WAITING) ||
        (state == EnumState_S_PROCESSING))
      aiPbMgrWaitAck();
  }

  return res;
}

static void aiPbMgrSetMetaInfo(const ai_buffer_meta_info *meta_info, const int idx,
    aiBufferShapeMsg *shape)
{
  shape->scale = 0.0f;
  shape->zeropoint = 0;
  if (AI_BUFFER_META_INFO_INTQ(meta_info)) {
    shape->scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(meta_info, idx);
    shape->zeropoint = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(meta_info, idx);
  }
}

bool aiPbMgrSendAiBuffer4(const reqMsg *req, respMsg *resp, EnumState state,
    uint32_t type, uint32_t id, ai_float dur_ms, const ai_buffer *buffer,
    ai_float scale, ai_i32 zero_point)
{
  struct aiPbMgrBuffer hdlb;
  const ai_buffer_meta_info *meta_info = AI_BUFFER_META_INFO(buffer);

#if defined(AI_PB_FULL_IO) && (AI_PB_FULL_IO == 1)
  const int is_io = AI_BUFFER_FMT_FLAG_IS_IO & buffer->format;
#endif

  hdlb.n_ops = 0;
  hdlb.buffer = (ai_buffer *)buffer;
  hdlb.err = EnumError_E_NONE;
  hdlb.n_max = aiPbAiBufferSize(buffer);
  hdlb.msg = NULL;

#if defined(AI_PB_FULL_IO) && (AI_PB_FULL_IO == 1)
  if ((type & PB_BUFFER_TYPE_SEND_WITHOUT_DATA) && (!is_io)) {
    hdlb.n_max  = 0;
  }
#else
  if (type & PB_BUFFER_TYPE_SEND_WITHOUT_DATA) {
    hdlb.n_max  = 0;
  }
#endif
  type &= (~PB_BUFFER_TYPE_SEND_WITHOUT_DATA);

  /* Fill Node sub-message */
  resp->which_payload = respMsg_node_tag;
  resp->payload.node.type = type;
  resp->payload.node.id = id;
  resp->payload.node.duration = dur_ms;
  ai_init_buffer_shape_msg(buffer, &resp->payload.node.buffer.shape);

  if (meta_info && scale == 0.0f)
    aiPbMgrSetMetaInfo(meta_info, 0, &resp->payload.node.buffer.shape);
  else {
    resp->payload.node.buffer.shape.scale = scale;
    resp->payload.node.buffer.shape.zeropoint = zero_point;
  }

  resp->payload.node.buffer.datas.funcs.encode = &aiPbBuffer_write_cb3;
  resp->payload.node.buffer.datas.arg = &hdlb;

  /* Send msg */
  aiPbMgrSendResp(req, resp, state);

  /* Waiting ACK */
  if (state == EnumState_S_PROCESSING)
    return aiPbMgrWaitAck();
  else
    return true;
}

/*---------------------------------------------------------------------------*/

void aiPbCmdSync(const reqMsg *req, respMsg *resp, void *param)
{
  resp->which_payload = respMsg_sync_tag;
  resp->payload.sync.version =
      EnumVersion_P_VERSION_MAJOR << 8 |
      EnumVersion_P_VERSION_MINOR;

  resp->payload.sync.capability = EnumCapability_CAP_FIXED_POINT;

#if defined(AI_PB_TEST) && (AI_PB_TEST == 1)
  resp->payload.sync.capability |= EnumCapability_CAP_SELF_TEST;
#endif

  resp->payload.sync.rtid = (uint32_t)param >> 16;
  resp->payload.sync.capability |= ((uint32_t)param & 0xFFFF);

  resp->payload.sync.rtid |= (_ARM_TOOLS_ID << 8);

  aiPbMgrSendResp(req, resp, EnumState_S_IDLE);
}

void aiPbCmdSysInfo(const reqMsg *req, respMsg *resp, void *param)
{
  UNUSED(param);
  resp->which_payload = respMsg_sinfo_tag;
  resp->payload.sinfo.devid = HAL_GetDEVID();
#ifdef STM32MP1
  resp->payload.sinfo.sclock = HAL_RCC_GetSystemCoreClockFreq();
  resp->payload.sinfo.hclock = HAL_RCC_GetHCLK3Freq();
#else
  resp->payload.sinfo.sclock = HAL_RCC_GetSysClockFreq();
  resp->payload.sinfo.hclock = HAL_RCC_GetHCLKFreq();
#endif
  resp->payload.sinfo.cache = getFlashCacheConf();

  aiPbMgrSendResp(req, resp, EnumState_S_IDLE);
}

static void init_aibuffer_msg(const ai_buffer *aibuffer, aiBufferShapeMsg *msg)
{
  if ((!aibuffer) || (!msg))
    return;

#if defined(TFLM_RUNTIME) && TFLM_RUNTIME == 1
  struct tflm_c_buffer *from_ = (struct tflm_c_buffer *)aibuffer;

  ai_init_buffer_shape_msg(aibuffer, msg);

  if (from_->buffer.meta_info) {
    msg->scale = from_->extra.scale;
    msg->zeropoint = from_->extra.zero_point;
  } else {
    msg->scale = 0.0f;
    msg->zeropoint = 0;
  }
#else
  const ai_buffer_meta_info *meta_info = AI_BUFFER_META_INFO(aibuffer);

  ai_init_buffer_shape_msg(aibuffer, msg);

  msg->addr = (uint32_t)aibuffer->data;
  aiPbMgrSetMetaInfo(meta_info, 0, msg);

#endif
}

static bool nn_shape_w_cb(pb_ostream_t *stream, const pb_field_t *field,
    const ai_buffer *aibuffer, int maxw)
{
  aiBufferShapeMsg msg;

  for (int i = 0; i < maxw; i++) {
    if (!pb_encode_tag_for_field(stream, field))
      return false;

    init_aibuffer_msg(&aibuffer[i], &msg);

    if (!pb_encode_submessage(stream, aiBufferShapeMsg_fields, &msg))
      return false;
  }
  return true;
}

static bool nn_inputs_w_cb(pb_ostream_t *stream, const pb_field_t *field,
    void * const *arg)
{
  ai_network_report *report = (ai_network_report *)*arg;

  if (!report)
    return true;

  return nn_shape_w_cb(stream, field, &report->inputs[0], report->n_inputs);
}

static bool nn_outputs_w_cb(pb_ostream_t *stream, const pb_field_t *field,
    void * const *arg)
{
  ai_network_report *report = (ai_network_report *)*arg;

  if (!report)
    return true;

  return nn_shape_w_cb(stream, field, &report->outputs[0], report->n_outputs);
}

void aiPbStrCopy(const char *src, char *dst, uint32_t max)
{
  const char undef[] = "UNDEFINED";
  size_t l = strlen(src);

  if (l > max)
    l = max-1;

  if (!dst)
    return;

  if (src && l)
    memcpy(dst, src, l+1);
  else
    memcpy(dst, undef, strlen(undef)+1);
}

uint32_t aiPbVersionToUint32(const ai_platform_version *ver)
{
  if (!ver)
    return 0;

  return ver->major << 24 | ver->minor << 16
      | ver->micro << 8 | ver->reserved;
}

void aiPbMgrSendNNInfo(const reqMsg *req, respMsg *resp,
    EnumState state, const ai_network_report *nn)
{
  resp->which_payload = respMsg_ninfo_tag;

  aiPbStrCopy(nn->model_name,
      &resp->payload.ninfo.model_name[0],
      sizeof(resp->payload.ninfo.model_name));
  aiPbStrCopy(nn->model_signature,
      &resp->payload.ninfo.model_signature[0],
      sizeof(resp->payload.ninfo.model_signature));
  aiPbStrCopy(nn->model_datetime,
      &resp->payload.ninfo.model_datetime[0],
      sizeof(resp->payload.ninfo.model_datetime));
  aiPbStrCopy(nn->compile_datetime,
      &resp->payload.ninfo.compile_datetime[0],
      sizeof(resp->payload.ninfo.compile_datetime));
  aiPbStrCopy(nn->runtime_revision,
      &resp->payload.ninfo.runtime_revision[0],
      sizeof(resp->payload.ninfo.runtime_revision));
  aiPbStrCopy(nn->tool_revision,
      &resp->payload.ninfo.tool_revision[0],
      sizeof(resp->payload.ninfo.tool_revision));

  resp->payload.ninfo.n_inputs = nn->n_inputs;
  resp->payload.ninfo.n_outputs = nn->n_outputs;
  resp->payload.ninfo.n_nodes = nn->n_nodes;
  resp->payload.ninfo.n_macc = nn->n_macc;

  resp->payload.ninfo.signature = nn->signature;
  resp->payload.ninfo.api_version =
      aiPbVersionToUint32(&nn->api_version);
  resp->payload.ninfo.interface_api_version =
      aiPbVersionToUint32(&nn->interface_api_version);
  resp->payload.ninfo.runtime_version =
      aiPbVersionToUint32(&nn->runtime_version);
  resp->payload.ninfo.tool_version =
      aiPbVersionToUint32(&nn->tool_version);
  resp->payload.ninfo.tool_api_version =
      aiPbVersionToUint32(&nn->tool_api_version);

  /* Minimum workaround to be compliant with MSG API 2.2 - can be updated in the future
   *  simple descriptor is provided.
   */
  ai_size size = 0;
#if defined(TFLM_RUNTIME) && TFLM_RUNTIME == 1
  size = AI_BUFFER_SHAPE_ELEM(&nn->activations, AI_SHAPE_CHANNEL);
#else
  if (nn->map_activations.size) {
    for (int i=0; i<nn->map_activations.size; i++)
      size += nn->map_activations.buffer[i].size;
    }
#endif
  ai_buffer tmp_ai_buff_act = AI_BUFFER_INIT(AI_FLAG_NONE, AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, size, 1, 1),
      size,
      NULL, NULL);
  init_aibuffer_msg(&tmp_ai_buff_act, &resp->payload.ninfo.activations);

  size = 0;
#if defined(TFLM_RUNTIME) && TFLM_RUNTIME == 1
  size = AI_BUFFER_SHAPE_ELEM(&nn->params, AI_SHAPE_CHANNEL);
#else
  if (nn->map_weights.size) {
    for (int i=0; i<nn->map_weights.size; i++)
      size += nn->map_weights.buffer[i].size;
  }
#endif
  ai_buffer tmp_ai_buff_w = AI_BUFFER_INIT(AI_FLAG_NONE, AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, size, 1, 1),
      size,
      NULL, NULL);
  init_aibuffer_msg(&tmp_ai_buff_w, &resp->payload.ninfo.weights);

  resp->payload.ninfo.inputs.funcs.encode = nn_inputs_w_cb;
  resp->payload.ninfo.inputs.arg = (void *)nn;

  resp->payload.ninfo.outputs.funcs.encode = nn_outputs_w_cb;
  resp->payload.ninfo.outputs.arg = (void *)nn;

  aiPbMgrSendResp(req, resp, state);
}


/*---------------------------------------------------------------------------*/

#if defined(AI_PB_TEST) && (AI_PB_TEST == 1)

#define AI_BUFFER_SHAPE_SET_ELEM(buf_, pos_, val_) AI_BUFFER_SET_SHAPE_ELEM(buf_, pos_, val_)
#define AI_BUFFER_SHAPE_GET_ELEM(buf_, pos_)       AI_BUFFER_SHAPE_ELEM(buf_, pos_)

#define _POS_WIDTH     AI_SHAPE_WIDTH
#define _POS_HEIGHT    AI_SHAPE_HEIGHT
#define _POS_CHANNELS  AI_SHAPE_CHANNEL
#define _POS_N_BATCHES AI_SHAPE_BATCH
#define _POS_DEPTH     AI_SHAPE_DEPTH
#define _POS_EXTENSION AI_SHAPE_EXTENSION

#include <stdlib.h>
#include <stdio.h>

#define _MAX_BUFF_SIZE_ (1024)
static ai_float buffer_test[_MAX_BUFF_SIZE_];
static ai_buffer ai_buffer_test =
    AI_BUFFER_INIT(AI_FLAG_NONE,                                                    /* flags */
        AI_BUFFER_FORMAT_FLOAT,                                                     /* format */
        AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 6, (1), (_MAX_BUFF_SIZE_), (1), (1), (1), (1)),   /* shape */
        (1) * (1) * AI_PAD_CHANNELS(AI_BUFFER_FORMAT_FLOAT, _MAX_BUFF_SIZE_),       /* size */
        NULL,                                                                       /* meta info */
        (ai_handle)buffer_test);                                                    /* data */

void aiPbTestRstAiBuffer(ai_buffer *buffer)
{
  ai_buffer_test.format = AI_BUFFER_FORMAT_FLOAT;
  ai_buffer_test.size = (1) * (1) * AI_PAD_CHANNELS(AI_BUFFER_FORMAT_FLOAT, _MAX_BUFF_SIZE_);
  AI_BUFFER_SHAPE_SET_ELEM(&ai_buffer_test, _POS_CHANNELS, _MAX_BUFF_SIZE_);
  AI_BUFFER_SHAPE_SET_ELEM(&ai_buffer_test, _POS_N_BATCHES, 1);
  AI_BUFFER_SHAPE_SET_ELEM(&ai_buffer_test, _POS_HEIGHT, 1);
  AI_BUFFER_SHAPE_SET_ELEM(&ai_buffer_test, _POS_WIDTH, 1);
  AI_BUFFER_SHAPE_SET_ELEM(&ai_buffer_test, _POS_EXTENSION, 1);
  AI_BUFFER_SHAPE_SET_ELEM(&ai_buffer_test, _POS_DEPTH, 1);
  ai_buffer_test.shape.size = 6;

  for (int i=0; i<_MAX_BUFF_SIZE_; i++) {
    ai_float value = 2.0f * (ai_float) rand() / (ai_float) RAND_MAX - 1.0f;
    buffer_test[i] = value;
  }
}

/* https://rosettacode.org/wiki/CRC-32#Python */
__STATIC_INLINE uint32_t rc_crc32(uint32_t crc, const char *buf, size_t len)
{
  static uint32_t table[256];
  static int have_table = 0;
  uint32_t rem;
  uint8_t octet;
  int i, j;
  const char *p, *q;

  /* This check is not thread safe; there is no mutex. */
  if (have_table == 0) {
    /* Calculate CRC table. */
    for (i = 0; i < 256; i++) {
      rem = i;  /* remainder from polynomial division */
      for (j = 0; j < 8; j++) {
        if (rem & 1) {
          rem >>= 1;
          rem ^= 0xedb88320;
        } else
          rem >>= 1;
      }
      table[i] = rem;
    }
    have_table = 1;
  }

  crc = ~crc;
  q = buf + len;
  for (p = buf; p < q; p++) {
    octet = *p;  /* Cast to unsigned octet. */
    crc = (crc >> 8) ^ table[(crc & 0xff) ^ octet];
  }
  return ~crc;
}

static uint32_t aiPbTestCalculateHash(const ai_buffer *aibuf)
{
  uint32_t crc = 0;
  if (!aibuf || !aibuf->data)
    return 0;

  const char *buf = (const char *)aibuf->data;

  size_t len = AI_BUFFER_BYTE_SIZE(AI_BUFFER_SIZE(aibuf), aibuf->format);
  crc = rc_crc32(0, buf, len);

  return crc;
}

/* TEST - Receive and/or Send a simple buffer */
static void aiPbTestCmdSimpleBuffer(const reqMsg *req, respMsg *resp,
    void *param)
{
  uint32_t sb = 0;
  uint32_t stest = req->param & 0xFFFF;
  uint32_t crc;
  uint32_t extra = req->param >> 16;

  UNUSED(param);

  aiPbTestRstAiBuffer(&ai_buffer_test);

  if (stest == 100)
  {
    /*
     * Normal/nominal processing operation (download buff only)
     *  ACK with expected buffer size
     *  Download the buffer
     *  Calculate hash
     *  Final ACK + hash value, DONE
     */
    ai_buffer *buff = &ai_buffer_test;

    ai_size nb_ch = AI_BUFFER_SHAPE_GET_ELEM(buff,_POS_CHANNELS);
    sb = (extra < nb_ch)?extra:nb_ch;

    /* Update format */
    AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_CHANNELS, sb);
    ai_buffer_test.shape.size = 4;

    if (req->opt)
      ai_buffer_test.format = aiPbMsgFmtToAiFmt(req->opt);

    ai_buffer_test.size = (1) * (1) * AI_PAD_CHANNELS(ai_buffer_test.format, sb);

    /* Send a ACK with the expected number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_WAITING, sb, EnumError_E_NONE);

    /* Wait & Receive the buffer */
    aiPbMgrReceiveAiBuffer3(req, resp, EnumState_S_PROCESSING,
        &ai_buffer_test);

    /* Calculate hash */
    crc = aiPbTestCalculateHash(&ai_buffer_test);

    /* Send ACK/DONE (operation DONE) */
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, crc, EnumError_E_NONE);
  }
  else if ((stest == 200) || (stest == 201))
  {
    /*
     * Normal/nominal - upload only
     *  ACK with expected buffer size
     *  Upload the buffer + op DONE
     */
    uint32_t ntype = 0;
    ai_buffer *buff = &ai_buffer_test;

    ai_buffer_test.shape.size = 4;

    /* Update format */
    if (stest == 200) {
      if (extra != 0) {
        ai_size nb_ch = AI_BUFFER_SHAPE_GET_ELEM(buff,_POS_CHANNELS);
        sb = (extra < nb_ch)?extra:nb_ch;
        AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_CHANNELS, sb);
      }
      else
      {
        ntype = PB_BUFFER_TYPE_SEND_WITHOUT_DATA;
        AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_CHANNELS, 32);
      }
    } else {
      /* stest = 201 */
      /* extra : b << 12, h << 8, w << 4, c << 0 */
      AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_N_BATCHES, extra >> 12 & 0xF);
      AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_HEIGHT, extra >> 8 & 0xF);
      AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_WIDTH, extra >> 4 & 0xF);
      AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_CHANNELS, extra & 0xF);
      sb = AI_BUFFER_SIZE(&ai_buffer_test);
    }

    if (req->opt)
      ai_buffer_test.format = aiPbMsgFmtToAiFmt(req->opt);

    /* Send a ACK with the expected number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, extra, EnumError_E_NONE);

    /* Send buffer and operation DONE */
    aiPbMgrSendAiBuffer4(req, resp, EnumState_S_DONE,
       ntype , 0, 0, &ai_buffer_test, 0.0f, 0);
  }
  else if (stest == 300)
  {
    /*
     * Normal/nominal - send/receive
     *  ACK with expected buffer size
     *  Download the buffer
     *  Upload the buffer
     */

    ai_buffer *buff = &ai_buffer_test;

    /* set the expected shape */
    if (req->opt) {
      const int e_ = req->opt >> 4 & 0xF;
      const int d_ = req->opt & 0xF;
      if (e_ == 0)
        ai_buffer_test.shape.size = 5;
      else
        ai_buffer_test.shape.size = 6;
      AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_EXTENSION, e_);
      AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_DEPTH, d_);
    } else {
      ai_buffer_test.shape.size = 4;
    }
    AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_N_BATCHES, extra >> 12 & 0xF);
    AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_HEIGHT, extra >> 8 & 0xF);
    AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_WIDTH, extra >> 4 & 0xF);
    AI_BUFFER_SHAPE_SET_ELEM(buff, _POS_CHANNELS, extra & 0xF);
    sb = AI_BUFFER_SIZE(buff) * AI_BUFFER_SHAPE_ELEM(buff, AI_SHAPE_BATCH);
    ai_buffer_test.size = sb;
    ai_buffer_test.format = AI_BUFFER_FORMAT_U8;

    /* Send a ACK with the expected number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, sb, EnumError_E_NONE);

    /* Wait & Receive the buffer */
    aiPbMgrReceiveAiBuffer3(req, resp, EnumState_S_PROCESSING,
        &ai_buffer_test);

    /* re-send buffer and operation DONE */
    aiPbMgrSendAiBuffer4(req, resp, EnumState_S_DONE,
       0 , 0, 0, &ai_buffer_test, 0.0f, 0);
  }
  else
  {
    aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
        EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
  }
}

/* TEST CMD */
void aiPbTestCmd(const reqMsg *req, respMsg *resp, void *param)
{
  uint32_t stest = req->param & 0xFFFF;
  if (stest == 0)
  {
    /* Send simple ACK: S_DONE/E_NONE */
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, 0, EnumError_E_NONE);
  }
  else if (stest == 1)
  {
    /* Send simple ACK msg: S_ERROR/E_INVALID_PARAM if no name is provided */
    if (!req->name[0]) {
      aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
          EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
    } else {
      aiPbMgrSendAck(req, resp, EnumState_S_DONE,
          strlen(req->name), EnumError_E_NONE);
    }
  }
  else if (stest == 2)
  {
    /* Send simple ACK msg: S_ERROR/E_INVALID_PARAM if no op */
    if (req->opt != 0xAABB) {
      aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
          EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
    } else {
      aiPbMgrSendAck(req, resp, EnumState_S_DONE,
          strlen(req->name), EnumError_E_NONE);
    }
  }
  else if (stest == 3)
  {
    /* Time out test */
    HAL_Delay(500);
  }
  else if (stest == 10)
  {
    /* 1 - Send ACK with provided ~opt : S_POCESSING/E_NONE
           delay
       2 - Send S_POCESSING/E_NONE
    */
    aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, ~req->opt,
        EnumError_E_NONE);
    HAL_Delay(10);
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, req->param,
        EnumError_E_NONE);
  }
  else if (stest == 20)
  {
      const char *str = "Hello..";
      aiPbMgrSendLog(req, resp, EnumState_S_DONE, 1, str);
  }
  else if ((stest >= 100) && (stest < 400))
  {
    aiPbTestCmdSimpleBuffer(req, resp, param);
  }
  else
  {
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, req->param + 1,
        EnumError_E_NONE);
  }
}

#endif
