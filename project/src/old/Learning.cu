#include "Learning.cuh"

__global__ void prepareBatch(
    unsigned char* __restrict__ lines,
    const unsigned int  lines_lenght,
    RESULT_TYPE* __restrict__ results,
    unsigned char* inputs,
    unsigned char* labels)
{
    unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;

    if (offset < lines_lenght)
    {
        unsigned int ol = offset * LINE_SIZE + ITEM_SIZE;
        unsigned int ob = offset * BATCH_SIZE;
        unsigned int or = offset * RESULT_VALUES + 1;
        for (unsigned int i = 0; i < LINE_SIZE; i++)
        {
            inputs[ob + i] = lines[ol + i]; //Url
            inputs[ob + i + ITEM_SIZE] = lines[ol + i + ITEM_SIZE]; //Agent
        }
        ob += 2 * ITEM_SIZE;
        inputs[ob] = results[or] > 0;    //user id
        inputs[ob+1] = results[or+7]>>8; //url len H
        inputs[ob+2] = results[or+7]&255;//url len L
        inputs[ob+3] = results[or+8]>>8; //agent len H
        inputs[ob+4] = results[or+8]&255;//agent len L
        labels[offset] = results[or +RESULT_VALUES - 2]; //category
    }
}


MySession* my_model_load(const char* filename, const char* input_name, const char* output_name) {
    printf("Loading model %s\n", filename);
    CStatus status;

    auto graph = tf_obj_unique_ptr(TF_NewGraph());
    {
        // Load a protobuf containing a GraphDef
        auto graph_def = tf_obj_unique_ptr(read_tf_buffer_from_file(filename));
        if (!graph_def) {
            return nullptr;
        }

        auto graph_opts = tf_obj_unique_ptr(TF_NewImportGraphDefOptions());
        TF_GraphImportGraphDef(graph.get(), graph_def.get(), graph_opts.get(), status.ptr);
    }

    if (status.failure()) {
        status.dump_error();
        return nullptr;
    }

    auto input_op = TF_GraphOperationByName(graph.get(), input_name);
    auto output_op = TF_GraphOperationByName(graph.get(), output_name);
    if (!input_op || !output_op) {
        return nullptr;
    }

    auto session = std::make_unique<MySession>();
    {
        auto opts = tf_obj_unique_ptr(TF_NewSessionOptions());
        session->session = tf_obj_unique_ptr(TF_NewSession(graph.get(), opts.get(), status.ptr));
    }

    if (status.failure()) {
        return nullptr;
    }
    assert(session);

    graph.swap(session->graph);
    session->inputs = { input_op, 0 };
    session->outputs = { output_op, 0 };

    return session.release();
}