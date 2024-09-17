package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

  	jwt "github.com/appleboy/gin-jwt/v2"
	"github.com/gin-gonic/gin"
)
  
func main() {

	var inc = func(x int) int { return x + 1}

    fmt.Println("Hello, World!")

	fmt.Println(inc(1))


	r := gin.Default()
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})
	r.Run() // listen and serve on 0.0.0.0:8080 (for windows "localhost:8080")
}


