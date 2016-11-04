function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Ray Tracer';
	UI.titleShort = 'RayTracerSimple';

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `RaytracingDemoFS - GL`,
		id: `RaytracingDemoFS`,
		initialValue: `

precision highp float;

struct PointLight {
  vec3 position;
  vec3 color;
};

struct Material {
  vec3  diffuse;
  vec3  specular;
  float glossiness;

  // Put the variables for reflection and refraction here
  
  /*
  	reflectionWeight
  	reflectionWeigh tells about the reflectivity of the material. Value between 0.0 and 1.0 with 
    0.0 for perfectly non reflective and 1.0 for perfect reflective surface.
  */
  float reflectionWeight; 
  
  /*
  	refIndex
    refIndex is the refractive index of material.
  */
  float refIndex;
  
  /*
  	refractionWeight
    This parameter is used to describe the level of refractivity of the material. Its intention is different 
    from refractive index. It can be compared to opacity and tells how much the surface will be reractive or 
    see-through. Its value is also between 0 and 1 
  */
  float refractionWeight;
};

struct Sphere {
  vec3 position;
  float radius;
  Material material;
};

struct Plane {
  vec3 normal;
  float d;
  Material material;
};

struct Cylinder {
  vec3 position;
  vec3 direction;  
  float radius;
  Material material;
};

const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 1;

struct Scene {
  vec3 ambient;
  PointLight[lightCount] lights;
  Sphere[sphereCount] spheres;
  Plane[planeCount] planes;
  Cylinder[cylinderCount] cylinders;
};

struct Ray {
  vec3 origin;
  vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
  bool hit;
  float t;
  vec3 position;
  vec3 normal;
  Material material;
};

HitInfo getEmptyHit() {
  return HitInfo(
    false, 
    0.0, 
    vec3(0.0), 
    vec3(0.0), 
  	// Depending on the material definition extension you make, this constructor call might need to be extened as well
    Material(vec3(0.0), vec3(0.0), 0.0,0.0,0.0,0.0)
    );
}

HitInfo intersectSphere(const Ray ray, const Sphere sphere) {
  
    vec3 to_sphere = ray.origin - sphere.position;
  
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, to_sphere);
    float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);
      	float t = min(t0, t1);
      	vec3 hitPosition = ray.origin + t * ray.direction;
        return HitInfo(
          	true,
          	t,
          	hitPosition,
          	normalize(hitPosition - sphere.position),
          	sphere.material);
    }
    return getEmptyHit();
}
/*
	---------------------------------------------------
    			Ray-Plane Intersection
    ---------------------------------------------------
	Equation of ray with origin at O and direction D is
	P = O + t*D
    Equation of plane at distance d and normal V is
  	P.V + d= 0
    For ray and plane intersection
    (O+t*D).V +d=0
    O.V + t*D.V + d = 0
    t= -(O.V +d)/(D.V)
    Point of intersection is (O +t*D )
    Normal at point of intersection is plane normal i.e V
    The material the we return in the hitInfo is the material of plane.

*/
  
HitInfo intersectPlane(const Ray ray,const Plane plane) {
  vec3  O = ray.origin;
  vec3  D = ray.direction;
  vec3  V = plane.normal;
  float d = plane.d;
  
  float DVdot = dot(D,V);
  float OVdot = dot(O,V);
  
  if(DVdot != 0.0){
  	float t= -(dot(O,V) + d)/ dot(D,V);
    vec3 hitPosition = O + t*D;
  	return HitInfo(
          	true,
          	t,
          	hitPosition,
          	plane.normal,
          	plane.material);
  }
  return getEmptyHit();
}

float lengthSquared(vec3 x) {
  return dot(x, x);
}

/*
	---------------------------------------------------
    			Ray-Cylinder Intersection
    ---------------------------------------------------
    Equation of ray with origin at O and direction D is
	P = O + t*D
    Equation of cylinder at position C, direction V and radius r
  	A = C +V*m , where m is a scalar that determines the closest 
    point on the axis to the hit point. 
    For ray and cylinder intersection
    (P-C-V*m) . V = 0
   	(P-C).V = m*(V.V) = m   (len(V)=1)
    (P-C).V = m
    (O + D*t -C).V =m
    Let X = O-C
    So
   	m = (D*t+X).V
   	m = D.V*t + X.V
    
    Also
    len(P-C-V*m) = r
    (D*t+X - V*(D.V*t + X.V)).((D*t+X - V*(D.V*t + X.V))) = r^2
    t^2(((D-V*(D.V))).((D-V*(D.V)))) 
    +2t((X-V*(X.V)).(D-V*(D.V)))
    + (X-V*(X.V)).(X-V*(X.V)) = r^2
    Simplification gives equation of form
    a*t^2+b*t+c=0
    Here
    a= D.D -D.V*D.V;
    b=2(D.X -(D.V)*(X.V))
    c=X.X -(X.V)*(X.V)-r^2
    
    We solve this quadratic equation and find solution of t which is min of t1 and t2
    d= b*b - 4*a*c
    
    Then the hit position is
    hitPosition = O + t * D
    The normal at the point of hit position is
    normal = normalise(hitPosition - C -V*m)
    the material the we return in the hitInfo is the material of cylinder.
*/

HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder) {
  vec3 O = ray.origin;
  vec3 D = ray.direction;
  vec3 C = cylinder.position;
  vec3 V = cylinder.direction;
  float r= cylinder.radius;
  vec3 X = O-C;
  float a = dot(D,D) -dot(D,V)*dot(D,V);
  float b = 2.0 * (dot(D, X)-dot(D,V)*dot(X,V));
  float c = dot(X, X) -dot(X,V)*dot(X,V) -r*r;
  float d = b * b - 4.0 * a * c;
  if (d > 0.0)
    {
		float t0 = (-b - sqrt(d)) / (2.0 * a);
		float t1 = (-b + sqrt(d)) / (2.0 * a);
      	float t = min(t0, t1);
      	vec3 hitPosition = O + t * D;
      	float m = dot(D,V)*t+dot(X,V);
        return HitInfo(
          	true,
          	t,
          	hitPosition,
          	normalize(hitPosition - C - V*m),
          	cylinder.material);
    }
    return getEmptyHit();
}

HitInfo intersectScene(const Scene scene, const Ray ray, float tMin, float tMax)
{
    HitInfo best_hit_info;
    best_hit_info.t = tMax;
  	best_hit_info.hit = false;

      for (int i = 0; i < cylinderCount; ++i) {
        Cylinder cylinder = scene.cylinders[i];
        HitInfo hit_info = intersectCylinder(ray, cylinder);

        if(	hit_info.hit && 
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

    for (int i = 0; i < sphereCount; ++i) {
        Sphere sphere = scene.spheres[i];
        HitInfo hit_info = intersectSphere(ray, sphere);

        if(	hit_info.hit && 
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

    for (int i = 0; i < planeCount; ++i) {
        Plane plane = scene.planes[i];
        HitInfo hit_info = intersectPlane(ray, plane);

        if(	hit_info.hit && 
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

  
  return best_hit_info;
}

vec3 shadeFromLight(
  const Scene scene,
  const Ray ray,
  const HitInfo hit_info,
  const PointLight light)
{ 
  vec3 hitToLight = light.position - hit_info.position;
  
  vec3 lightDirection = normalize(hitToLight);
  vec3 viewDirection = normalize(hit_info.position - ray.origin);
  vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);

  float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal));
  float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness);
  // Put your shadow test here
  
  /*
  To check whether a pixel is shadowed or not , we do a simple test of 
  shooting a shadow ray from the hitPoint towards light source and check if 
  the ray intersects any object in between the hitPoint and light source.
  If an intersection is detected, it means some object is obstructing the
  light to the the hitPoint and it is under shadow i.e dark else not 
  shadowded. If shadowed, we make the visibility of pixel as 0.
  Shadow Ray origin is taken as hitPoint position and direction is taken as
  d= lightPosition-hitPosition
 
  One common mistake that may occur and which should carefully be taken
  into consideration of is that only those intersection should be
  considered which occur between the hitPoint and light source. Any 
  intersection of shadowRay with any object behind the hitPoint i.e t<0
  or behind the light source i.e t > distance(lightSource and hitPoint)
  should not count towards shadow.
  */
  float visibility = 1.0;
  Ray shadowRay; // Shadow Ray
  shadowRay.origin = hit_info.position;
  shadowRay.direction = hitToLight;
  //Check if shadowRay intersects with any object in scene
  HitInfo shadowHitInfo = intersectScene(scene, shadowRay, 0.001, 100000.0);
  
  //Distance between light source and hitPoint
  float hitLightDist = distance(light.position,hit_info.position);
  if(shadowHitInfo.hit){
    //To check if intersection happened between light and hitPoint
    if(hit_info.t>0.0 && hit_info.t<=hitLightDist){
    	visibility=0.0;
    } 
  }
  
  return 	visibility * 
    		light.color * (
    		specular_term * hit_info.material.specular +
      		diffuse_term * hit_info.material.diffuse);
}


vec3 background(const Ray ray) {
  // A simple implicit sky that can be used for the background
  return vec3(0.1) + vec3(0.4, 0.5, 0.8) * max(0.0, ray.direction.y);
}

vec3 shade(const Scene scene, const Ray ray, const HitInfo hit_info) {
  
  	if(!hit_info.hit) {
  		return background(ray);
  	}
  
    vec3 shading = scene.ambient * hit_info.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
        shading += shadeFromLight(scene, ray, hit_info, scene.lights[i]); 
    }
    return shading;
}

Ray getFragCoordRay(const vec2 frag_coord) {
  	float sensorDistance = 1.0;
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax- sensorMin) / vec2(800, 400);
  	vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance));  
  
  	return Ray(origin, direction);
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
      
    Ray initialRay = getFragCoordRay(fragCoord);  
  	HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.001, 10000.0);  
  	vec3 result = shade(scene, initialRay, initialHitInfo);
	
  	Ray currentRay;
  	HitInfo currentHitInfo;
  	
  	// Compute the reflection
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;
  	// The initial strength of the reflection
  	float reflectionWeight = 1.0;

  	const int maxReflectionStepCount = 2;
  	for(int i = 0; i < maxReflectionStepCount; i++) {
      if(!currentHitInfo.hit) break;

      Ray nextRay;
	  // Put your code to compute the reflection ray
      nextRay.direction = reflect(currentRay.direction,currentHitInfo.normal);
      nextRay.origin= currentHitInfo.position;

      reflectionWeight = reflectionWeight*currentHitInfo.material.reflectionWeight;
      currentRay = nextRay;     

      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);            
      result += reflectionWeight * shade(scene, currentRay, currentHitInfo);
    }

  	// Compute the refraction
  	currentRay = initialRay;  
  	currentHitInfo = initialHitInfo;
  	// The initial strength of the refraction.
  	float refractionWeight = 1.0;

  	const int maxRefractionStepCount = 2;
  	for(int i = 0; i < maxRefractionStepCount; i++) {
       
      if(!currentHitInfo.hit) break;

      Ray nextRay;
	    // Put your code to compute the reflection ray
      nextRay.direction = 	refract(currentRay.direction,currentHitInfo.normal,1.0/currentHitInfo.material.refIndex);
      nextRay.origin= currentHitInfo.position;
	  refractionWeight = refractionWeight*currentHitInfo.material.refractionWeight;
      currentRay=nextRay;
      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
      result += refractionWeight * shade(scene, currentRay, currentHitInfo);
    }

  return result;
}

Material getDefaultMaterial() {
  return Material(vec3(0.3), vec3(0), 1.0,1.0,0.0,0.0);
}

Material getPaperMaterial() {
  // Replace by your definition of a paper material
  Material m;
  m.diffuse  =vec3(0.3, 0.3, 0.3);
  m.specular =vec3(0.0,0.0,0.0);
  m.glossiness=1.0;
  m.reflectionWeight=0.0;
  m.refIndex=1.9;
  m.refractionWeight =0.0;
  return m;
}

Material getPlasticMaterial() {
  // Replace by your definition of a plastic material
  Material m;
  m.diffuse  =vec3(1.0, 0.0, 0.0);
  m.specular =vec3(0.6,0.6,0.6);
  m.glossiness=20.0,
  m.reflectionWeight=0.1;
  m.refIndex=0.0;
  m.refractionWeight =0.0;
  return m;
}

Material getGlassMaterial() {
  // Replace by your definition of a glass material
  Material m;
  m.diffuse  =vec3(0.0, 0.0, 0.0);
  m.specular =vec3(0.0,0.0,0.0);
  m.glossiness=1.0;
  m.reflectionWeight=0.4;
  m.refIndex=1.3;
  m.refractionWeight =1.0;
  return m;
}

Material getSteelMirrorMaterial() {
  // Replace by your definition of a steel mirror material
  Material m;
  m.diffuse  =vec3(0.0);//,0.0,0.0);
  m.specular =vec3(0.5,0.5,0.5);
  m.glossiness=10.0;
  m.reflectionWeight=0.6;
  m.refIndex=0.0;
  m.refractionWeight =0.0;
  
  return m;}

void main()
{
    // Setup scene
    Scene scene;

  	scene.ambient = vec3(0.12, 0.15, 0.2);
  
    // Lights
    scene.lights[0].position = vec3(5, 15, -5);
    scene.lights[0].color    = 0.5 * vec3(0.8, 0.6, 0.5);

    scene.lights[1].position = vec3(-15, 10, 2);
    scene.lights[1].color    = 0.5 * vec3(0.7, 0.5, 1);

    // Primitives
    scene.spheres[0].position            	= vec3(6, -2, -12);
    scene.spheres[0].radius              	= 5.0;
    scene.spheres[0].material 				= getPaperMaterial();

    scene.spheres[1].position            	= vec3(-6, -2, -12);
    scene.spheres[1].radius             	= 4.0;
    scene.spheres[1].material				= getPlasticMaterial();

    scene.spheres[2].position            	= vec3(0, 2, -12);
    scene.spheres[2].radius              	= 3.0;
    scene.spheres[2].material   			= getGlassMaterial();

    scene.planes[0].normal            		= vec3(0, 1, 0);
  	scene.planes[0].d              			= 4.5;
    scene.planes[0].material				= getSteelMirrorMaterial();

    scene.cylinders[0].position            	= vec3(0, 2, -10);
  	scene.cylinders[0].direction            = normalize(vec3(1, 3, -1));
  	scene.cylinders[0].radius         		= 0.5;
    scene.cylinders[0].material				= getPaperMaterial();

  // compute color for fragment
  gl_FragColor.rgb = colorForFragment(scene, gl_FragCoord.xy);
  gl_FragColor.a = 1.0;
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
		title: `RaytracingDemoVS - GL`,
		id: `RaytracingDemoVS`,
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

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
	try {
		gl = canvas.getContext("experimental-webgl");
		gl.viewportWidth = canvas.width;
		gl.viewportHeight = canvas.height;
	} catch (e) {
	}
	if (!gl) {
		alert("Could not initialise WebGL, sorry :-(");
	}
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

    console.log(str);
	gl.shaderSource(shader, str);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		alert(gl.getShaderInfoLog(shader));
		return null;
	}

	return shader;
}

function RaytracingDemo() {
}

RaytracingDemo.prototype.initShaders = function() {

	this.shaderProgram = gl.createProgram();

	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoVS"));
	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoFS"));
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

RaytracingDemo.prototype.initBuffers = function() {
	this.triangleVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	
	var vertices = [
		 -1,  -1,  0,
		 -1,  1,  0,
		 1,  1,  0,

		 -1,  -1,  0,
		 1,  -1,  0,
		 1,  1,  0,
	 ];
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	this.triangleVertexPositionBuffer.itemSize = 3;
	this.triangleVertexPositionBuffer.numItems = 3 * 2;
}

RaytracingDemo.prototype.drawScene = function() {
			
	var perspectiveMatrix = new J3DIMatrix4();	
	perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

	var modelViewMatrix = new J3DIMatrix4();	
	modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);
		
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
	
	gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RaytracingDemo.prototype.run = function() {
	this.initShaders();
	this.initBuffers();

	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	gl.clear(gl.COLOR_BUFFER_BIT);

	this.drawScene();
};

function init() {	
	

	env = new RaytracingDemo();	
	env.run();

    return env;
}

function compute(canvas)
{
    env.initShaders();
    env.initBuffers();

    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    env.drawScene();
}
