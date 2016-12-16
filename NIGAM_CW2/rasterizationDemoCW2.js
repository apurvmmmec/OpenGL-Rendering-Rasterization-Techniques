function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Rasterization Demo';
	UI.titleShort = 'rasterizationDemo';

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `Rasterization`,
		id: `RasterizationDemoFS`,
		initialValue: `#define PROJECTION
#define RASTERIZATION
#define CLIPPING
#define INTERPOLATION
#define ZBUFFERING

precision highp float;

// Polygon / vertex functionality
const int MAX_VERTEX_COUNT = 8;

uniform ivec2 VIEWPORT;

struct Vertex {
    vec3 position;
    vec3 color;
};

struct Polygon {
    // Numbers of vertices, i.e., points in the polygon
    int vertexCount;
    // The vertices themselves
    Vertex vertices[MAX_VERTEX_COUNT];
};

// Appends a vertex to a polygon
void appendVertexToPolygon(inout Polygon polygon, Vertex element) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == polygon.vertexCount) {
            polygon.vertices[i] = element;
        }
    }
    polygon.vertexCount++;
}

// Copy Polygon source to Polygon destination
void copyPolygon(inout Polygon destination, Polygon source) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        destination.vertices[i] = source.vertices[i];
    }
    destination.vertexCount = source.vertexCount;
}

// Get the i-th vertex from a polygon, but when asking for the one behind the last, get the first again
Vertex getWrappedPolygonVertex(Polygon polygon, int index) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == polygon.vertexCount) return polygon.vertices[0];
        if (i == index) return polygon.vertices[i];
    }
}

// Creates an empty polygon
void makeEmptyPolygon(out Polygon polygon) {
  polygon.vertexCount = 0;
}

// Clipping part

#define ENTERING 0
#define LEAVING 1
#define OUTSIDE 2
#define INSIDE 3

int getCrossType(Vertex poli1, Vertex poli2, Vertex wind1, Vertex wind2) {
#ifdef CLIPPING
    // Put your code here
  return INSIDE;
#else
    return INSIDE;
#endif
}

// This function assumes that the segments are not parallel or collinear.
Vertex intersect2D(Vertex a, Vertex b, Vertex c, Vertex d) {
#ifdef CLIPPING
    // Put your code here
  	// We know that he equation of line betwen 2 points (x1,y1) and (x2,y2) is 
  	//              y-y1=( (y2-y1)/(x2-x1) )* (x-x1)
  	// or 			y= ( (y2-y1)/(x2-x1) )* (x-x1) + y1 ------------------ (1)
  	// Similarly another line can be wrien as
  	//				y= ( (y4-y3)/(x4-x3) )* (x-x3) + y3	------------------ (2)
  	// Solving eqn 1 an 2 , we can get x and y coord of intersection point
                       
  	float x1,y1,x2,y2,x3,y3,x4,y4,A1,A2,B1,B2,C1,C2,delta,x,y;
  	Vertex res;
  	x1 = a.position.x; y1 = a.position.y;
  	x2 = b.position.x; y2 = b.position.y;
  	x3 = c.position.x; y3 = c.position.y;
  	x4 = d.position.x; y4 = d.position.y;
  	
    A1 = y2-y1;
    A2 = y4-y3;
  	B1 = x1-x2;
    B2 = x3-x4;
  	
    C1 = A1*x1 + B1*y1;
  	C2 = A2*x3 + B2*y3;
    delta = A1*B2 - A2*B1;
  	
  	x = (B2*C1 - B1*C2)/delta;
	y = (A1*C2 - A2*C1)/delta;
  	res.position = vec3(x,y,0.0); // Assigning z as 0.0 because I am calculating the correct interpolated values of z in interpolation function
    return res;
  
#else
    return a;
#endif
}

#define INNER_SIDE 0
#define OUTER_SIDE 1

// Assuming a clockwise (vertex-wise) polygon, returns whether the input point 
// is on the inner or outer side of the edge (ab)
int edge(vec2 point, Vertex a, Vertex b) {
#ifdef RASTERIZATION
    // Put your code here
  // We know that he equation of line betwen 2 points (x1,y1) and (x2,y2) is 
  	//              y-y1=( (y2-y1)/(x2-x1) )* (x-x1)
  	//				(x2-x1)(y-y1) - (y2-y1)*(x-x1) = P
    // So for in point (x,y) on this line P=0 ,else depending on its on right or left of line , P > 0 or P < 0 
  float X=point.x;
  float Y=point.y;
  float x1 = a.position.x;
  float y1=a.position.y;
  float x2=b.position.x;
  float y2=b.position.y;
  
  float position = sign((x2 - x1) * (Y - y1) - (y2 - y1) * (X - x1));
  if (position == -1.0)
    	return -1;     // Return -1 of point is outside line
  else if(position== 1.0)
      return 1;        // Return +1, if point is inside line
  else
    return 0;			// Return 0 if point is on line

  #endif
    return OUTER_SIDE;
}

void sutherlandHodgmanClip(Polygon unClipped, Polygon clipWindow, out Polygon result) {
    Polygon clipped;
    copyPolygon(clipped, unClipped); //O/P=[a,b,c,d]
	Vertex A,B;
    // Loop over the clip window
  	// Define Edge E by selecting two vertices A and B from clipPolygon
  	// If A is 0th vertex B is last vertex
  	// Else if A is i vertex, B is (i-i) vertex
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i >= clipWindow.vertexCount) break;
    	
      	// Deal with all edges  of clipWindow one by one,
        // I have chosen edge of clipWindow in following order, 
        // If B ----> A is the edge of clipWindow, then I pick up edges in following order
        // last ----> 0, 0--->1,	1---2,	(last-1)--->last	
        A=clipWindow.vertices[i];
		if(i==0){
          B= getWrappedPolygonVertex(clipWindow,clipWindow.vertexCount-1)  ;
          }
       	  else{ // Choose B is (i-1)th vertex
                B=clipWindow.vertices[i-1];         	
       	  }      
		
      	// Now for every edge  B--->A of clipWindowm we do following tests for each edge of unclipped polygon.
        // Make a temporary copy of the current outputList polygon
        Polygon oldClipped;
        copyPolygon(oldClipped, clipped);

        // Set the outputList polygon to be empty
        makeEmptyPolygon(clipped);
      	
      	//We pick up the edges of unclipped polygon the same way as we picked for clipPloygon
      	// If B ----> A is the edge of clipWindow, then I pick up edges in following order
        // last ----> 0, 0--->1,	1---2,	(last-1)--->last	
		Vertex S = getWrappedPolygonVertex(oldClipped,oldClipped.vertexCount-1);  // last vertex
        // Loop over the inputList polygon
        for (int j = 0; j < MAX_VERTEX_COUNT; ++j) {
            if (j >= oldClipped.vertexCount) break;
            // Handle the j-th vertex of the outputList polygon. This should make use of the function 
            // intersect() to be implemented above.
#ifdef CLIPPING
            // Put your code here
          	Vertex E =  getWrappedPolygonVertex(oldClipped,j);
          	int e = edge(vec2(E.position.x,E.position.y),A,B);
            int s = edge(vec2(S.position.x,S.position.y),A,B);
          	if(e == 1){ //If Vertex E is inside
              if (s == -1){ //If vertex S is outside 
              	appendVertexToPolygon(clipped, intersect2D(S,E,A,B)); // Append the intersection point to clipped poly
              }
              appendVertexToPolygon(clipped, E); // Append the vertex that is inside i.e E  to clipped poly
            }
          	else if (s == 1){ // If vertex S is inside 
              appendVertexToPolygon(clipped, intersect2D(S,E,A,B)); // Append the intersection point to clipped poly
            }
          	S=E;  // For the next iteration S will be E and E will be incremented to next vertex
#else
            appendVertexToPolygon(clipped, getWrappedPolygonVertex(oldClipped, j));
#endif
        }
    }

    // Copy the last version to the output
    copyPolygon(result, clipped);
}

// Rasterization and culling part

// Returns if a point is inside a polygon or not
bool isPointInPolygon(vec2 point, Polygon polygon) {
    // Don't evaluate empty polygons
    if (polygon.vertexCount == 0) return false;
    // Check against each edge of the polygon
    bool rasterise = true;
  	int rPrev=-2;   // Variable to store the result of whether point inside or outside previous Edge 
  	int rCurrent=0; // Variable to store the result of whether point inside or outside current  Edge 
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#ifdef RASTERIZATION
            // Put your code here
          if(i==0){
             rCurrent = edge(point,polygon.vertices[i],getWrappedPolygonVertex(polygon,polygon.vertexCount-1));            
          }
       	  else{
            rCurrent = edge(point,polygon.vertices[i],polygon.vertices[i-1]);
       	  }
          if(rPrev == -2){ // Execution will come here only 1st time for every vertex. Here we set rPrev to rCurrent 
            rPrev = rCurrent;
          }
          else{
            if(rPrev*rCurrent > 0){ // point was inside for prev edge and also for current edge, rPrev*rCurrent>0 else <0. If point is inside for all edges, then its inside polygon
      			rasterise = true;
              	rPrev=rCurrent;
            }
  			else{
    			rasterise = false;
            	break;
            }
          }            
#else
            rasterise = false;
#endif
        }
    }
    return rasterise;
}

bool isPointOnPolygonVertex(vec2 point, Polygon polygon) {
    float pointSize = 0.008;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            if(length(polygon.vertices[i].position.xy - point) < pointSize) return true;
        }
    }
    return false;
}

float triangleArea(vec2 a, vec2 b, vec2 c) {
    // https://en.wikipedia.org/wiki/Heron%27s_formula
    float ab = length(a - b);
    float bc = length(b - c);
    float ca = length(c - a);
    float s = (ab + bc + ca) / 2.0;
    return sqrt(max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
}

Vertex interpolateVertex(vec2 point, Polygon polygon) {
    float weightSum = 0.0;
    vec3 colorSum = vec3(0.0);
    vec3 positionSum = vec3(0.0);
    float depthSum = 0.0;
  	Vertex result;
  	mat2 zBuffer;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#if defined(INTERPOLATION) || defined(ZBUFFERING)
          // Put your code here
          
          // Temporary Vertex variables to hold values of vertices opposite to current i'th vertex 
          Vertex V1,V2; 
          
          V1 = getWrappedPolygonVertex(polygon,i+1);
          
          // Check the case when (i+2) becomes more than total vertex count
          if(i+1 == polygon.vertexCount){ 
            V2 = getWrappedPolygonVertex(polygon,1);

          }else{
            V2 = getWrappedPolygonVertex(polygon,i+2);
          }
          
          //Weight for current vertex equals area of triangle formed by the point and vertices opposite to current vertex.
          float weight = triangleArea(vec2(V1.position.x,V1.position.y) ,vec2(V2.position.x,V2.position.y),point); 
          
          // Add weights to calculate total area of traingle 
          weightSum = weightSum + weight; 
          
          // Calculate depthSum from weighted 1/z from every vertex
          depthSum = depthSum+weight*(1.0/polygon.vertices[i].position.z); 

#else
#endif
#ifdef ZBUFFERING
            // Put your code here     
          
          //Calculate interpolated vertex co-ordinates
          positionSum= positionSum+weight*polygon.vertices[i].position; 
#endif
#ifdef INTERPOLATION
          
          // Calculate interpolated vertex color
          colorSum= colorSum + weight*polygon.vertices[i].color; 
          
#endif
        }
    }
    
    //Vertex result = polygon.vertices[0];
  
#ifdef INTERPOLATION
    colorSum /= weightSum;
    positionSum /= weightSum;
    depthSum /= weightSum;
    colorSum /= depthSum;    
    result.color = colorSum;
#endif
#ifdef ZBUFFERING
    positionSum /= depthSum;
    result.position = positionSum;
#endif

  return result;
}

// Projection part

// Used to generate a projection matrix.
mat4 computeProjectionMatrix() {
    mat4 projectionMatrix = mat4(1);

#ifdef PROJECTION
    // Put your code here
  // We use the simplest projection matrix for this exercise as the fov, aspect ratio , near and far planes are not given
  projectionMatrix = mat4(1,0,0,0,
                          0,1,0,0,
                          0,0,1,1,
                          0,0,0,1);
#endif
  
    return projectionMatrix;
}

// Used to generate a simple "look-at" camera. 
mat4 computeViewMatrix(vec3 VRP, vec3 TP, vec3 VUV) {
    mat4 viewMatrix = mat4(1);
  

#ifdef PROJECTION
    // Put your code here
  	// ViewPositionNormal = target Position - View Referenc3 Position
    vec3 VPN= TP-VRP;
  	//Calculate u,v and n. Note that u,v,n reperesent the view co-ordinate system axes and are orthogonal to each other
  	vec3 n = normalize(VPN);
  	vec3 u = normalize(cross(VUV,n));
  	vec3 v = cross(n,u);
  	
  	//The last row of view matrix takes care of the camera translation form (0,0,0) to VRP
  	viewMatrix = mat4(u.x,v.x,n.x,0.0,
                      u.y,v.y,n.y,0.0,
                      u.z,v.z,n.z,0.0,
                      -dot(VRP,u),-dot(VRP,v),-dot(VRP,n),1.0);
#endif
    return viewMatrix;
}

// Takes a single input vertex and projects it using the input view and projection matrices
vec3 projectVertexPosition(vec3 position) {

  // Set the parameters for the look-at camera.
    vec3 TP =  vec3(0, 0, 0);
    vec3 VRP = vec3(0, 0, -7);
    vec3 VUV = vec3(0, 1, 0);
  
    // Compute the view matrix.
    mat4 viewMatrix = computeViewMatrix(VRP, TP, VUV);

  // Compute the projection matrix.
    mat4 projectionMatrix = computeProjectionMatrix();
  
#ifdef PROJECTION
    // Put your code here
  // Applying the view and projection transform to the input vertex which ahs to be projected
   vec4 p2 = projectionMatrix*viewMatrix*vec4(position,1.0);
  // Performing the perspective divide. This step (x/w,y/w,z/w,w/w) make the 4th component of 
  // homogenous coordinate system 1 again (x',y',z',1) so (x',y',z') forms the projected vertex.
  return vec3(p2.x/p2.w,p2.y/p2.w,p2.z/p2.w); 				
#else
    return position;
#endif
}

// Projects all the vertices of a polygon
void projectPolygon(inout Polygon projectedPolygon, Polygon polygon) {
    copyPolygon(projectedPolygon, polygon);
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            projectedPolygon.vertices[i].position = projectVertexPosition(polygon.vertices[i].position);
        }
    }
}

// Draws a polygon by projecting, clipping, ratserizing and interpolating it
void drawPolygon(
  vec2 point, 
  Polygon clipWindow, 
  Polygon oldPolygon, 
  inout vec3 color, 
  inout float depth)
{
    Polygon projectedPolygon;
    projectPolygon(projectedPolygon, oldPolygon);  
  
    Polygon clippedPolygon;
    sutherlandHodgmanClip(projectedPolygon, clipWindow, clippedPolygon);

    if (isPointInPolygon(point, clippedPolygon)) {
      
        Vertex interpolatedVertex = 
          interpolateVertex(point, projectedPolygon);
          
        if (interpolatedVertex.position.z < depth) {
            color = interpolatedVertex.color;
            depth = interpolatedVertex.position.z;
        }
    } else {
        if (isPointInPolygon(point, projectedPolygon)) {
            color = vec3(0.1, 0.1, 0.1);
        }
    }
  
   if (isPointOnPolygonVertex(point, clippedPolygon)) {
        color = vec3(1);
   }
}

// Main function calls

void drawScene(vec2 point, inout vec3 color) {
    color = vec3(0.3, 0.3, 0.3);
    point = vec2(2.0 * point.x / float(VIEWPORT.x) - 1.0, 2.0 * point.y / float(VIEWPORT.y) - 1.0);

    Polygon clipWindow;
    clipWindow.vertices[0].position = vec3(-0.750,  0.750, 1.0);
    clipWindow.vertices[1].position = vec3( 0.750,  0.750, 1.0);
    clipWindow.vertices[2].position = vec3( 0.750, -0.750, 1.0);
    clipWindow.vertices[3].position = vec3(-0.750, -0.750, 1.0);
    clipWindow.vertexCount = 4;
    color = isPointInPolygon(point, clipWindow) ? vec3(0.5, 0.5, 0.5) : color;

    const int triangleCount = 2;
    Polygon triangles[triangleCount];
  
    triangles[0].vertices[0].position = vec3(-7.7143, -3.8571, 1.0);
    triangles[0].vertices[1].position = vec3(7.7143, 8.4857, 1.0);
    triangles[0].vertices[2].position = vec3(4.8857, -0.5143, 1.0);
    triangles[0].vertices[0].color = vec3(1.0, 0.5, 0.1);
    triangles[0].vertices[1].color = vec3(0.2, 0.8, 0.2);
    triangles[0].vertices[2].color = vec3(0.2, 0.3, 1.0);
    triangles[0].vertexCount = 3;
  
    triangles[1].vertices[0].position = vec3(3.0836, -4.3820, 1.9);
    triangles[1].vertices[1].position = vec3(-3.9667, 0.7933, 0.5);
    triangles[1].vertices[2].position = vec3(-4.3714, 8.2286, 1.0);
    triangles[1].vertices[1].color = vec3(0.1, 0.5, 1.0);
    triangles[1].vertices[2].color = vec3(1.0, 0.6, 0.1);
    triangles[1].vertices[0].color = vec3(0.2, 0.6, 1.0);
    triangles[1].vertexCount = 3;

    float depth = 10000.0;
    // Project and draw all the triangles
    for (int i = 0; i < triangleCount; i++) {
        drawPolygon(point, clipWindow, triangles[i], color, depth);
    }   
}

void main() {
    drawScene(gl_FragCoord.xy, gl_FragColor.rgb);
    gl_FragColor.a = 1.0;
}`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: true,
		type: `text/javascript`,
		title: `Resolution settings`,
		id: `ResolutionJS`,
		initialValue: `// This variable sets the inverse scaling factor at which the rendering happens.
// The higher the constant, the faster it will be. SCALING = 1 is regular, non-scaled rendering.
SCALING = 1;`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RasterizationDemoTextureVS - GL`,
		id: `RasterizationDemoTextureVS`,
		initialValue: `attribute vec3 position;
    attribute vec2 textureCoord;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    varying highp vec2 vTextureCoord;
  
    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        vTextureCoord = textureCoord;
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RasterizationDemoVS - GL`,
		id: `RasterizationDemoVS`,
		initialValue: `attribute vec3 position;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-fragment`,
		title: `RasterizationDemoTextureFS - GL`,
		id: `RasterizationDemoTextureFS`,
		initialValue: `
        varying highp vec2 vTextureCoord;

        uniform sampler2D uSampler;

        void main(void) {
            gl_FragColor = texture2D(uSampler, vec2(vTextureCoord.s, vTextureCoord.t));
        }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
    try {
        gl = canvas.getContext("webgl");
        gl.viewportWidth = canvas.width;
        gl.viewportHeight = canvas.height;
    } catch (e) {
    }
    if (!gl) {
        alert("Could not initialise WebGL, sorry :-(");
    }
}

function evalJS(id) {
    var jsScript = document.getElementById(id);
    eval(jsScript.innerHTML);
}

function getShader(gl, id) {
    var shaderScript = document.getElementById(id);
    if (!shaderScript) {
        return null;
    }

    var str = "";
    var k = shaderScript.firstChild;
    while (k) {
        if (k.nodeType == 3) {
            str += k.textContent;
        }
        k = k.nextSibling;
    }

    var shader;
    if (shaderScript.type == "x-shader/x-fragment") {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    } else if (shaderScript.type == "x-shader/x-vertex") {
        shader = gl.createShader(gl.VERTEX_SHADER);
    } else {
        return null;
    }

    gl.shaderSource(shader, str);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }

    return shader;
}

function RasterizationDemo() {
}

RasterizationDemo.prototype.initShaders = function() {

    this.shaderProgram = gl.createProgram();

    gl.attachShader(this.shaderProgram, getShader(gl, "RasterizationDemoVS"));
    gl.attachShader(this.shaderProgram, getShader(gl, "RasterizationDemoFS"));
    gl.linkProgram(this.shaderProgram);

    if (!gl.getProgramParameter(this.shaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }

    gl.useProgram(this.shaderProgram);

    this.shaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.shaderProgram, "position");
    gl.enableVertexAttribArray(this.shaderProgram.vertexPositionAttribute);

    this.shaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.shaderProgram, "projectionMatrix");
    this.shaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.shaderProgram, "modelViewMatrix");
}

RasterizationDemo.prototype.initTextureShaders = function() {

    this.textureShaderProgram = gl.createProgram();

    gl.attachShader(this.textureShaderProgram, getShader(gl, "RasterizationDemoTextureVS"));
    gl.attachShader(this.textureShaderProgram, getShader(gl, "RasterizationDemoTextureFS"));
    gl.linkProgram(this.textureShaderProgram);

    if (!gl.getProgramParameter(this.textureShaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }

    gl.useProgram(this.textureShaderProgram);

    this.textureShaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.textureShaderProgram, "position");
    gl.enableVertexAttribArray(this.textureShaderProgram.vertexPositionAttribute);

    this.textureShaderProgram.textureCoordAttribute = gl.getAttribLocation(this.textureShaderProgram, "textureCoord");
    gl.enableVertexAttribArray(this.textureShaderProgram.textureCoordAttribute);
    //gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, 2, gl.FLOAT, false, 0, 0);

    this.textureShaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.textureShaderProgram, "projectionMatrix");
    this.textureShaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.textureShaderProgram, "modelViewMatrix");
}

RasterizationDemo.prototype.initBuffers = function() {
    this.triangleVertexPositionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
    
    var vertices = [
         -1.0,  -1.0,  0.0,
         -1.0,   1.0,  0.0,
          1.0,   1.0,  0.0,

         -1.0,  -1.0,  0.0,
          1.0,  -1.0,  0.0,
          1.0,   1.0,  0.0,
     ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    this.triangleVertexPositionBuffer.itemSize = 3;
    this.triangleVertexPositionBuffer.numItems = 3 * 2;

    this.textureCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);

    var textureCoords = [
        0.0,  0.0,
        0.0,  1.0,
        1.0,  1.0,

        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoords), gl.STATIC_DRAW);
    this.textureCoordBuffer.itemSize = 2;
}

RasterizationDemo.prototype.initTextureFramebuffer = function() {
    // create off-screen framebuffer
    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    this.framebuffer.width = this.prerender_width;
    this.framebuffer.height = this.prerender_height;

    // create RGB texture
    this.framebufferTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.framebufferTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.framebuffer.width, this.framebuffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);//LINEAR_MIPMAP_NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    //gl.generateMipmap(gl.TEXTURE_2D);

    // create depth buffer
    this.renderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.renderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.framebuffer.width, this.framebuffer.height);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.framebufferTexture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.renderbuffer);

    // reset state
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

RasterizationDemo.prototype.drawScene = function() {
            
    gl.bindFramebuffer(gl.FRAMEBUFFER, env.framebuffer);
    gl.useProgram(this.shaderProgram);
    gl.viewport(0, 0, this.prerender_width, this.prerender_height);
    gl.clear(gl.COLOR_BUFFER_BIT);

        var perspectiveMatrix = new J3DIMatrix4();  
        perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

        var modelViewMatrix = new J3DIMatrix4();    
        modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);

        gl.uniform2iv(gl.getUniformLocation(this.shaderProgram, "VIEWPORT"), [this.prerender_width, this.prerender_height]);
            
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
        gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, this.textureCoordBuffer.itemSize, gl.FLOAT, false, 0, 0);
        
        gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.textureShaderProgram);
    gl.viewport(0, 0, this.render_width, this.render_height);
    gl.clear(gl.COLOR_BUFFER_BIT);

        var perspectiveMatrix = new J3DIMatrix4();  
        perspectiveMatrix.setUniform(gl, this.textureShaderProgram.projectionMatrixUniform, false);

        var modelViewMatrix = new J3DIMatrix4();    
        modelViewMatrix.setUniform(gl, this.textureShaderProgram.modelviewMatrixUniform, false);

        gl.bindTexture(gl.TEXTURE_2D, this.framebufferTexture);
        gl.uniform1i(gl.getUniformLocation(this.textureShaderProgram, "uSampler"), 0);
            
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, this.textureCoordBuffer.itemSize, gl.FLOAT, false, 0, 0);
        
        gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RasterizationDemo.prototype.run = function() {
    evalJS("ResolutionJS");

    this.render_width     = 800;
    this.render_height    = 400;

    this.prerender_width  = this.render_width / SCALING;
    this.prerender_height = this.render_height / SCALING;

    this.initTextureFramebuffer();
    this.initShaders();
    this.initTextureShaders();
    this.initBuffers();
};

function init() {   
    env = new RasterizationDemo();

    return env;
}

function compute(canvas)
{
    env.run();
    env.drawScene();
}
