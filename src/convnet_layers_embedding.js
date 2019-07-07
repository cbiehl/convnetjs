(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  var EmbeddingLayer = function(opt) {

    var opt = opt || {};

    // required
		this.embedding_dim = opt.embedding_dim;
    this.in_depth = opt.in_depth;
		this.num_inputs = opt.in_depth;
    this.in_sx = 1;
    this.in_sy = 1;

    // optional
		if(opt.embedding_matrix){
			this.embedding_matrix = opt.embedding_matrix;
		}else{
			this.embedding_matrix = {};
		}

    // computed
    this.out_depth = this.in_depth;
    this.out_sx = this.embedding_dim;
    this.out_sy = 1;
    this.layer_type = 'embedding';
  }

  EmbeddingLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

			for(var i=0;i<this.in_depth;i++){
				var e = this.embedding_matrix[V.get(0,0,i)];
				if(e){
					for(var j=0;j<e.w.length;j++){
						A.set(j,0,i,e[j]);
					}
				}else{
					// out of vocabulary token
					if(is_training){
						// add token with random embedding to vocabulary
						var new_embedding = global.zeros(self.embedding_dim);
						for(var j=0;j<self.embedding_dim;j++){
							new_embedding[j] = global.randn(-1.0, 1.0);
							A.set(j,0,i,new_embedding[j]);
						}

						self.embedding_matrix[V.get(0,0,i)] = new Vol(new_embedding);

					}else{
						// produce random embedding
						for(var j=0;j<this.embedding_dim;j++){
							A.set(j,0,i,global.randn(-1.0, 1.0)); // TODO: better oov handling
						}
					}
				}
			}

      this.out_act = A;
      return this.out_act;
    },
    backward: function() {
			var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol

      // compute gradient wrt weights only (previous layer assumed to be input)
      for(var i=0;i<this.out_depth;i++) {
        for(var d=0;d<this.embedding_dim;d++) {
					this.embedding_matrix[i].dw[d] += this.out_act.dw[i];
        }
      }
    },
    getParamsAndGrads: function() {
			var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.embedding_matrix[i].w, grads: this.embedding_matrix[i].dw});
      }

      return response;
    },
    toJSON: function() {
      var json = {};
			json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
			json.num_inputs = this.num_inputs;
      json.layer_type = this.layer_type;
			json.embedding_dim = this.embedding_dim;
			json.embedding_matrix = this.embedding_matrix;

      return json;
    },
    fromJSON: function(json) {
			this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
			this.num_inputs = json.num_inputs;
      this.layer_type = json.layer_type;
			this.embedding_dim = json.embedding_dim;
			this.embedding_matrix = json.embedding_matrix;
    }
  }

  global.EmbeddingLayer = EmbeddingLayer;

})(convnetjs);
