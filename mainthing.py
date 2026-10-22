class Value : 
    def __init__ (self, data, _children = (), op = '' ) : 
        self.data = data 
        self.grad =  0 
        self._backward = lambda : None 
        self._prev = set ( _children ) 
        self._op = op 

    def __add__ ( self, other ) : 
        other = other if isinstance ( other, Value ) else Value ( other ) 
        out = Value ( self.data + other.data, (self, other ) , '+' ) 

        def _backward () : 
            self.grad += out.grad 
            other.grad += out.grad 

        out._backward = _backward 

        return out 

    def __mul__ ( self, other ) : 
        other = other if isinstance ( other, Value ) else Value ( other  ) 
        out = Value ( self.data * other.data , (self, other ) , '*' ) 

        def _backward () : 
            self.grad += out.grad * other.grad 
            other.grad += out.grad * self.grad 

        out._backward   = _backward 

        return out 

    def __pow__ (self, other ) : 
        assert isinstance ( other, (int, float ) )  
        out = Value ( self ** other, (self, other ) , '**' )   

        def _backward () : 
           self.grad += other * self.data ** ( other - 1 ) 

        out._backward = _backward 

        return out 
    
    def relu ( self ) : 
        out = Value ( max (0, self.data ), (self, ), 'relu' ) 
        
        def _backward () : 
            if out.data > 0 : 
                self.grad = out.grad 
        out._backward = _backward 

        return out 

    def backward (self ) : 
        topo = [] 
        visited = set () 
        def buildtop ( v ) : 
            if v not in visited : 
                visited.add ( v ) 
                for child in v._prev : 
                    buildtop (child ) 
                topo.append ( v ) 

        self.grad = 1 
        for v in reversed ( topo ) : 
            v._backward () 
         








